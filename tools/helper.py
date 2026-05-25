# import os, sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, "../"))

# import torch
# import time
# import numpy as np

# # def network_forward_train(base_model, regressor, pred_scores, video_1, label_1, video_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, batch_idx, batch_num, args):
# #     loss = 0.0
# #     start = time.time()
# #     combined_feature_1, combined_feature_2 = base_model(video_1, video_2, label=[label_1, label_2], is_train=True, theta=args.score_range)
# #     combined_feature = torch.cat((combined_feature_1, combined_feature_2), 0)
# #     out_prob, delta = regressor(combined_feature)
# #     glabel_1, rlabel_1 = group.produce_label(label_2 - label_1)
# #     glabel_2, rlabel_2 = group.produce_label(label_1 - label_2)
# #     leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
# #     leaf_probs_1 = leaf_probs[:leaf_probs.shape[0]//2]
# #     leaf_probs_2 = leaf_probs[leaf_probs.shape[0]//2:]
# #     delta_1 = delta[:delta.shape[0]//2]
# #     delta_2 = delta[delta.shape[0]//2:]
# #     loss += nll(leaf_probs_1, glabel_1.argmax(0))
# #     loss += nll(leaf_probs_2, glabel_2.argmax(0))
# #     for i in range(group.number_leaf()):
# #         mask = rlabel_1[i] >= 0
# #         if mask.sum() != 0:
# #             loss += mse(delta_1[:,i][mask].reshape(-1,1).float(), rlabel_1[i][mask].reshape(-1,1).float())
# #         mask = rlabel_2[i] >= 0
# #         if mask.sum() != 0:
# #             loss += mse(delta_2[:,i][mask].reshape(-1,1).float(), rlabel_2[i][mask].reshape(-1,1).float())
# #     loss.backward()

# #     if opti_flag:
# #         optimizer.step()
# #         optimizer.zero_grad()

# #     end = time.time()
# #     batch_time = end - start
# #     if batch_idx % args.print_freq == 0:
# #         print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
# #                 % (epoch, args.max_epoch, batch_idx, batch_num,
# #                 batch_time, loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

# #     # relative_scores = group.inference(leaf_probs_2.detach().cpu().numpy(), delta_2.detach().cpu().numpy())
# #     # if args.benchmark == 'Hei':
# #     #     score = relative_scores.cuda() + label_2
# #     #
# #     # pred_scores.extend([i.item() for i in score])
# #     # evaluate result of training phase
# #     relative_scores = group.inference(leaf_probs_2.detach().cpu().numpy(), delta_2.detach().cpu().numpy())
# #     if args.benchmark == 'MTL':
# #         if args.usingDD:
# #             score = (relative_scores.cuda() + label_2) * diff
# #         else:
# #             score = relative_scores.cuda() + label_2
# #     elif args.benchmark == 'Seven':
# #         score = relative_scores.cuda() + label_2
# #     elif args.benchmark == 'JIG':
# #         score = relative_scores.cuda() + label_2
# #     elif args.benchmark == 'Hei':
# #         score = relative_scores.cuda() + label_2
# #     else:
# #         raise NotImplementedError()
# #     pred_scores.extend([i.item() for i in score])

# # # def network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args):
# # #     if not video_2_list:
# # #         print(f"Warning: video_2_list is empty for video_1")
# # #         pred_scores.extend([0.0] * video_1.size(0))
# # #         return
# # #     score = torch.zeros(video_1.size(0), 1, device=video_1.device)
# # #     for video_2, label_2 in zip(video_2_list, label_2_list):
# # #         combined_feature = base_model(video_1, video_2, label=[label_2], is_train=False, theta=args.score_range)
# # #         out_prob, delta = regressor(combined_feature)
# # #         leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
# # #         relative_scores = group.inference(leaf_probs.detach().cpu().numpy(), delta.detach().cpu().numpy())
# # #         if args.benchmark == 'MTL':
# # #             if args.usingDD:
# # #                 score += (relative_scores.cuda() + label_2) * diff
# # #             else:
# # #                 score += relative_scores.cuda() + label_2
# # #         elif args.benchmark == 'Seven':
# # #             score += relative_scores.cuda() + label_2
# # #         elif args.benchmark == 'JIG':
# # #             score += relative_scores.cuda() + label_2
# # #         elif args.benchmark == 'Hei':
# # #             score += relative_scores.cuda() + label_2
# # #         else:
# # #             raise NotImplementedError()
# # #     avg_score = score / len(video_2_list)
# # #     pred_scores.extend(avg_score.detach().cpu().numpy().flatten())

# def dae_loss(mu, log_var, target):  # target是ground truth delta或相对分数
#     sigma_sq = torch.exp(log_var)
#     L_rec = (target - mu).pow(2) / sigma_sq  # 重建损失
#     L_sup = log_var  # 支持损失 (log(σ²))
#     alpha = 0.6
#     beta = 0.4
#     loss = alpha * L_rec.mean() + beta * L_sup.mean()  # 平均过batch和叶节点
#     return loss

# def network_forward_train(base_model, regressor, pred_scores, video_1, label_1, video_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, batch_idx, batch_num, args, duration_1=None, duration_2=None):
#     loss = 0.0
#     start = time.time()
    
#     # combined_feature_1, combined_feature_2, Rhy_1, Rhy_2 = base_model(video_1, video_2, is_train=True, label=[label_1, label_2], theta=args.score_range, duration_1=duration_1, duration_2=duration_2)
#     combined_feature_1, combined_feature_2, Rhy_1, Rhy_2, mu_1, log_var_1, mu_2, log_var_2 = base_model(
#         video_1, video_2, is_train=True, label=[label_1, label_2], theta=args.score_range, duration_1=duration_1, duration_2=duration_2
#     )
    
#     combined_feature = torch.cat((combined_feature_1, combined_feature_2), 0)
#     # out_prob, delta = regressor(combined_feature)
#     out_prob, delta, mu, log_var = regressor(combined_feature, is_train=True)  # Updated to receive mu and log_var
#     glabel_1, rlabel_1 = group.produce_label(label_2 - label_1)
#     glabel_2, rlabel_2 = group.produce_label(label_1 - label_2)
#     leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
#     leaf_probs_1 = leaf_probs[:leaf_probs.shape[0]//2]
#     leaf_probs_2 = leaf_probs[leaf_probs.shape[0]//2:]
#     delta_1 = delta[:delta.shape[0]//2]
#     delta_2 = delta[delta.shape[0]//2:]
#     loss += nll(leaf_probs_1, glabel_1.argmax(0))
#     loss += nll(leaf_probs_2, glabel_2.argmax(0))
#     # for i in range(group.number_leaf()):
#     #     mask = rlabel_1[i] >= 0
#     #     if mask.sum() != 0:
#     #         loss += mse(delta_1[:,i][mask].reshape(-1,1).float(), rlabel_1[i][mask].reshape(-1,1).float())
#     #     mask = rlabel_2[i] >= 0
#     #     if mask.sum() != 0:
#     #         loss += mse(delta_2[:,i][mask].reshape(-1,1).float(), rlabel_2[i][mask].reshape(-1,1).float())

#     # Replace MSE with DAE loss
#     for i in range(group.number_leaf()):
#         mask_1 = rlabel_1[i] >= 0
#         if mask_1.sum() != 0:
#             loss += dae_loss(
#                 mu_1[:, i][mask_1].reshape(-1, 1).float(),
#                 log_var_1[:, i][mask_1].reshape(-1, 1).float(),
#                 rlabel_1[i][mask_1].reshape(-1, 1).float()
#             )
#         mask_2 = rlabel_2[i] >= 0
#         if mask_2.sum() != 0:
#             loss += dae_loss(
#                 mu_2[:, i][mask_2].reshape(-1, 1).float(),
#                 log_var_2[:, i][mask_2].reshape(-1, 1).float(),
#                 rlabel_2[i][mask_2].reshape(-1, 1).float()
#             )

#     loss.backward()

#     if opti_flag:
#         optimizer.step()
#         optimizer.zero_grad()

#     end = time.time()
#     batch_time = end - start
#     if batch_idx % args.print_freq == 0:
#         print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
#               % (epoch, args.max_epoch, batch_idx, batch_num,
#                  batch_time, loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

#     relative_scores = group.inference(leaf_probs_2.detach().cpu().numpy(), delta_2.detach().cpu().numpy())
#     if args.benchmark == 'JIG':
#         score = relative_scores.cuda() + label_2
#     elif args.benchmark == 'Hei':
#         score = relative_scores.cuda() + label_2
#     else:
#         raise NotImplementedError()
#     pred_scores.extend([i.item() for i in score])



# # def network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args):
# #     print(f"Starting network_forward_test with video_1 shape: {video_1.shape}, len(video_2_list): {len(video_2_list)}")
# #     if not video_2_list:
# #         print("Warning: video_2_list is empty")
# #         pred_scores.extend([0.0] * video_1.size(0))
# #         return
# #     score = torch.zeros(video_1.size(0), 1, device=video_1.device)
# #     for i, (video_2, label_2) in enumerate(zip(video_2_list, label_2_list)):
# #         # print(f"Processing reference video {i+1}/{len(video_2_list)}")
# #         start_time = time.time()
# #         combined_feature = base_model(video_1, video_2, label=[label_2], is_train=False, theta=args.score_range)
# #         # print(f"Base model forward took {time.time() - start_time:.2f} seconds")
# #         out_prob, delta = regressor(combined_feature)
# #         # print(f"Regressor forward took {time.time() - start_time:.2f} seconds")
# #         leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
# #         relative_scores = group.inference(leaf_probs.detach().cpu().numpy(), delta.detach().cpu().numpy())
# #         if args.benchmark == 'Hei':
# #             score += relative_scores.cuda() + label_2
# #         elif args.benchmark == 'JIG':
# #             score += relative_scores.cuda() + label_2
# #             # score += misc.denormalize(relative_scores.cuda() + label_2, args.class_idx, args.score_range)
# #         else:
# #             raise NotImplementedError()

# #     avg_score = score / len(video_2_list)
# #     pred_scores.extend(avg_score.detach().cpu().numpy().flatten())
# #     print("Finished network_forward_test")
# def network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args, group_idx_1=None, group_idx_2_list=None):
#     print(f"Starting network_forward_test with video_1 shape: {video_1.shape}, len(video_2_list): {len(video_2_list)}")
#     if not video_2_list:
#         print("Warning: video_2_list is empty")
#         pred_scores.extend([0.0] * video_1.size(0))
#         return
#     score = torch.zeros(video_1.size(0), 1, device=video_1.device)
#     for i, (video_2, label_2, group_idx_2) in enumerate(zip(video_2_list, label_2_list, group_idx_2_list)):
#         start_time = time.time()
#         combined_feature, Rhy_1, Rhy_2 = base_model(video_1, video_2, is_train=False, label=[label_2], theta=args.score_range, duration_1=group_idx_1, duration_2=group_idx_2)
#         # out_prob, delta = regressor(combined_feature)
#         out_prob, delta, mu, log_var = regressor(combined_feature, is_train=False)  # Receive mu and log_var (unused in test)
#         leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
#         relative_scores = group.inference(leaf_probs.detach().cpu().numpy(), delta.detach().cpu().numpy())
#         if args.benchmark == 'JIG':
#             score += relative_scores.cuda() + label_2
#         elif args.benchmark == 'Hei':
#             score += relative_scores.cuda() + label_2
#         else:
#             raise NotImplementedError()
#     avg_score = score / len(video_2_list)
#     pred_scores.extend(avg_score.detach().cpu().numpy().flatten())
#     print("Finished network_forward_test")

# def save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, exp_name, args):
#     torch.save({
#                 'base_model': base_model.state_dict(),
#                 'regressor': regressor.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'epoch_best': epoch_best,
#                 'rho_best': rho_best,
#                 'L2_min': L2_min,
#                 'RL2_min': RL2_min,
#                 }, os.path.join(args.experiment_path, exp_name + '.pth'))

# def save_outputs(pred_scores, true_scores, args):
#     save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
#     save_path_true = os.path.join(args.experiment_path, 'true.npy')
#     np.save(save_path_pred, pred_scores)
#     np.save(save_path_true, true_scores)

# 修改
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import time
import numpy as np

def dae_loss(mu, log_var, target):
    sigma_sq = torch.exp(log_var)
    L_rec = (target - mu).pow(2) / sigma_sq  # 重建损失
    L_sup = log_var  # 支持损失 (log(σ²))
    alpha = 0.6
    beta = 0.4
    loss = alpha * L_rec.mean() + beta * L_sup.mean()  # 平均过batch和叶节点
    return loss

def network_forward_train(base_model, regressor, pred_scores, video_1, label_1, video_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, batch_idx, batch_num, args, duration_1=None, duration_2=None, scale_loss=1.0):
    loss = 0.0
    start = time.time()
    
    combined_feature_1, combined_feature_2, Rhy_1, Rhy_2, mu_1, log_var_1, mu_2, log_var_2 = base_model(
        video_1, video_2, True, label=[label_1, label_2], theta=args.score_range, duration_1=duration_1, duration_2=duration_2
    )
    
    combined_feature = torch.cat((combined_feature_1, combined_feature_2), 0)
    out_prob, delta, mu, log_var = regressor(combined_feature, True)
    glabel_1, rlabel_1 = group.produce_label(label_2 - label_1)
    glabel_2, rlabel_2 = group.produce_label(label_1 - label_2)
    
    # 调试信息：打印形状
    # print(f"mu_2 shape: {mu_2.shape}, log_var_2 shape: {log_var_2.shape}, rlabel_2 shape: {rlabel_2.shape}")
    
    leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
    leaf_probs_1 = leaf_probs[:leaf_probs.shape[0]//2]
    leaf_probs_2 = leaf_probs[leaf_probs.shape[0]//2:]
    delta_1 = delta[:delta.shape[0]//2]
    delta_2 = delta[delta.shape[0]//2:]
    mu_1 = mu[:mu.shape[0]//2]
    mu_2 = mu[mu.shape[0]//2:]
    log_var_1 = log_var[:log_var.shape[0]//2]
    log_var_2 = log_var[log_var.shape[0]//2:]
    
    loss += nll(leaf_probs_1, glabel_1.argmax(0))
    loss += nll(leaf_probs_2, glabel_2.argmax(0))
    
    # Replace MSE with DAE loss
    num_leaf = group.number_leaf()
    for i in range(num_leaf):
        mask_1 = rlabel_1[i] >= 0 if i < rlabel_1.shape[0] else torch.zeros_like(mu_1[:, 0], dtype=torch.bool)
        if mask_1.sum() != 0:
            loss += dae_loss(
                mu_1[:, i][mask_1].reshape(-1, 1).float(),
                log_var_1[:, i][mask_1].reshape(-1, 1).float(),
                rlabel_1[i][mask_1].reshape(-1, 1).float()
            )
        mask_2 = rlabel_2[i] >= 0 if i < rlabel_2.shape[0] else torch.zeros_like(mu_2[:, 0], dtype=torch.bool)
        if mask_2.sum() != 0:
            loss += dae_loss(
                mu_2[:, i][mask_2].reshape(-1, 1).float(),
                log_var_2[:, i][mask_2].reshape(-1, 1).float(),
                rlabel_2[i][mask_2].reshape(-1, 1).float()
            )

    # 梯度累积：缩放loss
    loss = loss * scale_loss
    loss.backward()

    if opti_flag:
        optimizer.step()
        optimizer.zero_grad()

    end = time.time()
    batch_time = end - start
    if batch_idx % args.print_freq == 0:
        print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f'
              % (epoch, args.max_epoch, batch_idx, batch_num,
                 batch_time, loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

    relative_scores = group.inference(leaf_probs_2.detach().cpu().numpy(), delta_2.detach().cpu().numpy())
    if args.benchmark == 'JIG':
        score = relative_scores.cuda() + label_2
    elif args.benchmark == 'Hei':
        score = relative_scores.cuda() + label_2
    else:
        raise NotImplementedError()
    pred_scores.extend([i.item() for i in score])
    
    # 显存优化：释放不需要的中间变量（batch_size=1时特别有用）
    try:
        del combined_feature_1, combined_feature_2, combined_feature
        del leaf_probs, leaf_probs_1, leaf_probs_2
        del delta_1, delta_2, mu_1, mu_2, log_var_1, log_var_2
        del mu, log_var, out_prob, delta
        del Rhy_1, Rhy_2
        del relative_scores, score
    except:
        pass  # 如果删除失败，继续执行

def network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args, group_idx_1=None, group_idx_2_list=None):
    print(f"Starting network_forward_test with video_1 shape: {video_1.shape}, len(video_2_list): {len(video_2_list)}")
    if not video_2_list:
        print("Warning: video_2_list is empty")
        pred_scores.extend([0.0] * video_1.size(0))
        return
    score = torch.zeros(video_1.size(0), 1, device=video_1.device)
    for i, (video_2, label_2, group_idx_2) in enumerate(zip(video_2_list, label_2_list, group_idx_2_list)):
        start_time = time.time()
        combined_feature, Rhy_1, Rhy_2 = base_model(video_1, video_2, False, label=[label_2], theta=args.score_range, duration_1=group_idx_1, duration_2=group_idx_2)
        out_prob, delta, mu, log_var = regressor(combined_feature, False)
        leaf_probs = out_prob[-1].reshape(combined_feature.shape[0], -1)
        relative_scores = group.inference(leaf_probs.detach().cpu().numpy(), delta.detach().cpu().numpy())
        if args.benchmark == 'JIG':
            score += relative_scores.cuda() + label_2
        elif args.benchmark == 'Hei':
            score += relative_scores.cuda() + label_2
        else:
            raise NotImplementedError()
    avg_score = score / len(video_2_list)
    pred_scores.extend(avg_score.detach().cpu().numpy().flatten())
    print("Finished network_forward_test")

def save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, exp_name, args):
    torch.save({
                'base_model': base_model.state_dict(),
                'regressor': regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'epoch_best': epoch_best,
                'rho_best': rho_best,
                'L2_min': L2_min,
                'RL2_min': RL2_min,
                }, os.path.join(args.experiment_path, exp_name + '.pth'))

def save_outputs(pred_scores, true_scores, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred.npy')
    save_path_true = os.path.join(args.experiment_path, 'true.npy')
    np.save(save_path_pred, pred_scores)
    np.save(save_path_true, true_scores)