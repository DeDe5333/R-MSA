import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc
import time
from sklearn.metrics import mean_absolute_error

def get_duration_group_idx(durations, num_groups=6):
    durations = np.array(durations)
    sorted_indices = np.argsort(durations)
    group_size = len(durations) // num_groups
    group_indices = np.zeros_like(durations, dtype=int)
    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < num_groups - 1 else len(durations)
        group_indices[sorted_indices[start:end]] = i
    return torch.tensor(group_indices, dtype=torch.float)

def test_net(args):
    print('Tester start ... ')
    train_dataset, test_dataset = builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                            shuffle=False,num_workers = int(args.workers),
                                            pin_memory=True)
    base_model, regressor = builder.model_builder(args)
    # load checkpoints
    builder.load_model(base_model, regressor, args)

    # if using RT, build a group
    group = builder.build_group(train_dataset, args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True

    #  DP
    base_model = nn.DataParallel(base_model)
    regressor = nn.DataParallel(regressor)

    test(base_model, regressor, test_dataloader, group, args)

def run_net(args):
    print('Trainer start ... ')
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    # 显存优化：batch_size=1时，关闭pin_memory，减少num_workers
    # pin_memory=False 可以减少显存使用，但可能略微降低数据传输速度
    # num_workers=0 或 1 可以避免多进程导致的显存累积
    use_pin_memory = args.bs_train > 1  # 只有batch_size>1时才使用pin_memory
    num_workers = max(0, min(1, int(args.workers))) if args.bs_train == 1 else int(args.workers)
    
    # 显存优化：强制DataLoader的batch_size=1（避免显存问题）
    # 但可以通过gradient_accumulation_steps模拟更大的batch_size
    dataloader_batch_size = 1  # 始终使用1，避免显存问题
    effective_batch_size = args.bs_train  # 这是期望的batch_size
    gradient_accumulation_steps = effective_batch_size // dataloader_batch_size
    
    print(f"显存优化模式：DataLoader batch_size={dataloader_batch_size}, "
          f"期望batch_size={effective_batch_size}, "
          f"梯度累积步数={gradient_accumulation_steps}")
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=dataloader_batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            pin_memory=False, worker_init_fn=misc.worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                            shuffle=False, num_workers=num_workers,
                                            pin_memory=use_pin_memory, drop_last=True)
    # build model
    base_model, regressor = builder.model_builder(args)

    # if using RT, build a group
    group = builder.build_group(train_dataset, args)
    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, regressor, args)

    # parameter setting
    start_epoch = 0
    global epoch_best, rho_best, mae_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    mae_best = float('inf')  # <--- 新增这行！初始化为正无穷
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best, rho_best, L2_min, RL2_min = \
            builder.resume_train(base_model, regressor, optimizer, args)
        print('resume ckpts @ %d epoch( rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (start_epoch - 1, rho_best,  L2_min, RL2_min))

    #  DP
    base_model = nn.DataParallel(base_model)
    regressor = nn.DataParallel(regressor)

    # loss
    mse = nn.MSELoss().cuda()
    nll = nn.NLLLoss().cuda()

    # trainval

    # training
    for epoch in range(start_epoch, args.max_epoch):
        true_scores = []
        pred_scores = []
        num_iter = 0
        base_model.train()  # set model to training mode
        regressor.train()
        if args.fix_bn:
            base_model.apply(misc.fix_bn)  # fix bn
        for idx, (data , target) in enumerate(train_dataloader):
            # break
            num_iter += 1
            
            true_scores.extend(data['final_score'].numpy())
            # data preparing
            # video_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'JIG':
                video_1 = data['video'].float().cuda()  # N, C, T, H, W
                label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                video_2 = target['video'].float().cuda()
                label_2 = target['final_score'].float().reshape(-1, 1).cuda()
                diff = None  # JIG 无 difficulty
                duration_1 = data['duration'].float()
                duration_2 = target['duration'].float()
                group_idx_1 = data['group'].float().cuda()
                group_idx_2 = target['group'].float().cuda()
            elif args.benchmark == 'Hei':
                video_1 = data['video'].float().cuda()  # N, C, T, H, W
                label_1 = data['final_score'].float().reshape(-1, 1).cuda()
                video_2 = target['video'].float().cuda()
                label_2 = target['final_score'].float().reshape(-1, 1).cuda()
                diff = None  
                duration_1 = data['duration'].float()
                duration_2 = target['duration'].float()
                group_idx_1 = data['group'].float().cuda()
                group_idx_2 = target['group'].float().cuda()
            else:
                raise NotImplementedError()
            
            # 梯度累积：只有当累积到足够的步数时才更新
            opti_flag = (num_iter % gradient_accumulation_steps == 0)
            # 调整loss的缩放（因为batch_size=1，所以loss不需要缩放）
            scale_loss = 1.0 / gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1.0

            # helper.network_forward_train(base_model, regressor, pred_scores, video_1, label_1, video_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, idx+1, len(train_dataloader), args)
            helper.network_forward_train(base_model, regressor, pred_scores, video_1, label_1, video_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, idx+1, len(train_dataloader), args, duration_1=group_idx_1, duration_2=group_idx_2, scale_loss=scale_loss)
            
            # 显存优化：每个batch后清理显存
            try:
                del video_1, video_2, label_1, label_2
                if args.benchmark in ['JIG', 'Hei']:
                    del group_idx_1, group_idx_2
                if diff is not None:
                    del diff
                # 清理data和target的引用
                del data, target
                torch.cuda.empty_cache()  # 清理未使用的显存缓存
            except:
                pass  # 如果删除失败，继续执行

        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        mae = mean_absolute_error(true_scores, pred_scores)
        L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
        print('[Training] EPOCH: %d, correlation: %.4f, MAE: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f'%(epoch, rho, mae, L2, RL2, optimizer.param_groups[0]['lr'],  optimizer.param_groups[1]['lr']))


        validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args)
        helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, 'last', args)
        print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f, best MAE: %.6f'%(epoch, rho_best, L2_min, RL2_min, mae_best))
        # scheduler lr
        if scheduler is not None:
            scheduler.step(L2)

# def validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args):
#     print("Start validating epoch {}".format(epoch))
#     global use_gpu
#     global epoch_best, rho_best, L2_min, RL2_min
#     true_scores = []
#     pred_scores = []
#     base_model.eval()  # set model to eval mode
#     regressor.eval()
#     batch_num = len(test_dataloader)
#     with torch.no_grad():
#         datatime_start = time.time()
#         for batch_idx,  (data , target) in enumerate(test_dataloader, 0):
#             datatime = time.time() - datatime_start
#             start = time.time()
#             true_scores.extend(data['final_score'].numpy())
#             # data prepare
#             if args.benchmark == 'MTL':
#                 video_1 = data['video'].float().cuda() # N, C, T, H, W
#                 if args.usingDD:
#                     label_2_list = [item['completeness'].float().reshape(-1,1).cuda() for item in target]
#                 else:
#                     label_2_list = [item['final_score'].float().reshape(-1,1).cuda() for item in target]
#                 diff = data['difficulty'].float().reshape(-1,1).cuda()
#                 video_2_list = [item['video'].float().cuda() for item in target]
#                 # check
#                 if not args.dive_number_choosing and args.usingDD:
#                     for item in target:
#                         assert (diff == item['difficulty'].float().reshape(-1,1).cuda()).all()
#             elif args.benchmark == 'Seven':
#                 video_1 = data['video'].float().cuda() # N, C, T, H, W
#                 video_2_list = [item['video'].float().cuda() for item in target]
#                 label_2_list = [item['final_score'].float().reshape(-1,1).cuda() for item in target]
#                 diff = None
#             elif args.benchmark == 'JIG':
#                 video_1 = data['video'].float().cuda()  # N, C, T, H, W
#                 video_2_list = [item['video'].float().cuda() for item in target]
#                 label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
#                 diff = None
#             elif args.benchmark == 'Hei':
#                 video_1 = data['video'].float().cuda()  # N, C, T, H, W
#                 video_2_list = [item['video'].float().cuda() for item in target]
#                 label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
#                 diff = None
#             else:
#                 raise NotImplementedError()
#             helper.network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args)
#             batch_time = time.time() - start
#             if batch_idx % args.print_freq == 0:
#                 print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
#                     % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
#             datatime_start = time.time()
#         # analysis on results
#         pred_scores = np.array(pred_scores)
#         true_scores = np.array(true_scores)
#         rho, p = stats.spearmanr(pred_scores, true_scores)
#         L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
#         RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
#         if L2_min > L2:
#             L2_min = L2
#         if RL2_min > RL2:
#             RL2_min = RL2
#         if rho > rho_best:
#             rho_best = rho
#             epoch_best = epoch
#             print('-----New best found!-----')
#             helper.save_outputs(pred_scores, true_scores, args)
#             helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, 'best', args)

#         print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f'%(epoch, rho, L2, RL2))

def validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best, rho_best, mae_best, L2_min, RL2_min
    true_scores = []
    pred_scores = []
    base_model.eval()
    regressor.eval()
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()
        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            if args.benchmark == 'JIG':
                video_1 = data['video'].float().cuda()
                video_2_list = [item['video'].float().cuda() for item in target]
                label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                group_idx_1 = data['group'].float().cuda()
                group_idx_2_list = [item['group'].float().cuda() for item in target]
                diff = None
                helper.network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args, group_idx_1=group_idx_1, group_idx_2_list=group_idx_2_list)
            elif args.benchmark == 'Hei':
                video_1 = data['video'].float().cuda()
                video_2_list = [item['video'].float().cuda() for item in target]
                label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                group_idx_1 = data['group'].float().cuda()
                group_idx_2_list = [item['group'].float().cuda() for item in target]
                diff = None
                helper.network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args, group_idx_1=group_idx_1, group_idx_2_list=group_idx_2_list)
            else:
                raise NotImplementedError()
            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
                    % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        mae = mean_absolute_error(true_scores, pred_scores)
        L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if mae_best > mae:
            mae_best = mae
        # if rho > rho_best:
        #     rho_best = rho
        #     epoch_best = epoch
        #     print('-----New best found!-----')
        #     helper.save_outputs(pred_scores, true_scores, args)
        #     helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, 'best', args)
        
        if rho > rho_best or (rho >= rho_best and mae < mae_best):  # rho优先，相同rho时选MAE更小的
            rho_best = rho
            # mae_best = mae
            epoch_best = epoch
            print('-----New best found!-----')
            helper.save_outputs(pred_scores, true_scores, args)
            helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, 'best', args)
        print('[TEST] EPOCH: %d, correlation: %.6f, MAE: %.4f, L2: %.6f, RL2: %.6f'%(epoch, rho, mae, L2, RL2))




def test(base_model, regressor, test_dataloader, group, args):
    global use_gpu
    true_scores = []
    pred_scores = []
    base_model.eval()
    regressor.eval()
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()
        for batch_idx, (data, target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            # data prepare
            if args.benchmark == 'JIG':
                video_1 = data['video'].float().cuda()
                video_2_list = [item['video'].float().cuda() for item in target]
                label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                group_idx_1 = data['group'].float().cuda()
                group_idx_2_list = [item['group'].float().cuda() for item in target]
                diff = None
            elif args.benchmark == 'Hei':
                video_1 = data['video'].float().cuda()
                video_2_list = [item['video'].float().cuda() for item in target]
                label_2_list = [item['final_score'].float().reshape(-1, 1).cuda() for item in target]
                group_idx_1 = data['group'].float().cuda()
                group_idx_2_list = [item['group'].float().cuda() for item in target]
                diff = None
            else:
                raise NotImplementedError()
            helper.network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args, group_idx_1=group_idx_1, group_idx_2_list=group_idx_2_list)
            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
                    % (batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f'%(rho, L2, RL2))
