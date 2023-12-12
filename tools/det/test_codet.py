import argparse
import os
from copy import deepcopy
from CMiMC.utils import AverageMeter

import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from CMiMC.datasets import V2XSimDet
from CMiMC.configs import Config, ConfigGlobal
from CMiMC.utils.CoDetModule import *
from CMiMC.utils.loss import *
from CMiMC.utils.mean_ap import eval_map
from CMiMC.models.det import *
from CMiMC.utils.detection_util import late_fusion
from CMiMC.utils.data_util import apply_pose_noise


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


@torch.no_grad()
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker
    apply_late_fusion = args.apply_late_fusion
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i
    MMI_flag = args.MMI_flag
    flag_GPU = args.flag_GPU

    # Specify gpu device
    if flag_GPU == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 0
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # GPU 1
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.com == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com":
        flag = "when2com"
        if args.inference == "argmax_test":
            flag = "who2com"
        if args.warp_flag:
            flag = flag + "_warp"
    elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
        flag = args.com
    elif args.com == "lowerbound":
        flag = "lowerbound"
        if args.box_com:
            flag += "_box_com"
    else:
        raise ValueError(f"com: {args.com} is not supported")

    print("flag", flag)
    config.flag = flag
    config.split = "test"

    # num_agent = args.num_agent
    num_agent = 5 if "v1" in args.logpath else 6
    # agent0 is the RSU
    agent_idx_range = range(num_agent) if args.rsu else range(1, num_agent)
    validation_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="upperbound" if args.com == "upperbound" else "lowerbound",
        kd_flag=args.kd_flag,
        rsu=args.rsu,
    )
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(validation_dataset))

    if not args.rsu:
        num_agent -= 1

    if flag == "upperbound" or flag.startswith("lowerbound"):
        model = FaFNet(
            config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent
        )
    elif flag.startswith("when2com") or flag.startswith("who2com"):
        model = When2com(
            config,
            layer=args.layer,
            warp_flag=args.warp_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )
    elif args.com == "disco":
        model = DiscoNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
            MMI_flag=MMI_flag
        )
    elif args.com == "v2v":
        model = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
            compress_level=compress_level,
            only_v2i=only_v2i,
        )

    if flag_GPU == 0:
        model = nn.DataParallel(model,device_ids=[0]) # GPU 1
    else:
        model = nn.DataParallel(model,device_ids=[1]) # GPU 1
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag,MMI_flag)

    if "MMI" not in args.resume:
        args.resume = args.resume.replace(args.com,
                                        args.com + ("+MMI" if MMI_flag else "") )
    model_save_path = args.resume[: args.resume.rfind("/")]

    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")

    os.makedirs(model_save_path, exist_ok=True)
    log_file_name = os.path.join(model_save_path, "log.txt")
    saver = open(log_file_name, "a")
    saver.write("\nGPU number: {}\n".format(torch.cuda.device_count()))
    saver.flush()

    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    checkpoint = torch.load(
        args.resume, map_location="cpu"
    )  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"],strict=False)
    # fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    if not apply_late_fusion:
        save_fig_path = [ # logs/flag/rsu/test(val)/epoch_x/visk/
            os.path.join(
                model_save_path,
                "val" if args.data.endswith("val") else "test",
                "epoch_{}".format(checkpoint["epoch"]),
                f"vis{i}") for i in agent_idx_range
        ]
        tracking_path = [ #  logs/flag/rsu/epoch_x/trackingk/
            os.path.join(
                model_save_path,
                "val" if args.data.endswith("val") else "test",
                "epoch_{}".format(checkpoint["epoch"]),
                f"tracking{i}") for i in agent_idx_range
        ]
    else:
        save_fig_path = [ # logs/flag/rsu/test(val)/epoch_x/visk/
            os.path.join(
                model_save_path,
                "val" if args.data.endswith("val_late") else "test_late",
                "epoch_{}".format(checkpoint["epoch"]),
                f"vis{i}") for i in agent_idx_range
        ]
        tracking_path = [ #  logs/flag/rsu/epoch_x/trackingk/
            os.path.join(
                model_save_path,
                "val" if args.data.endswith("val_late") else "test_late",
                "epoch_{}".format(checkpoint["epoch"]),
                f"tracking{i}") for i in agent_idx_range
        ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]

    running_loss_class = AverageMeter(
        "classification Loss", ":.6f"
    )  # for cell classification error
    running_loss_loc = AverageMeter(
        "Localization Loss", ":.6f"
    )  # for state estimation error

    tracking_file = [set()] * num_agent
    for cnt, sample in enumerate(validation_data_loader):
        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        print(filenames)



        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)

        # add pose noise
        if pose_noise > 0:
            apply_pose_noise(pose_noise, trans_matrices)

        if not args.rsu:
            num_all_agents -= 1

        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            "trans_matrices": trans_matrices.to(device),
        }

        if flag == "lowerbound_box_com":
            loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
                data, data["trans_matrices"]
            )
        elif flag == "disco":
            (
                loss,
                cls_loss,
                loc_loss,
                result,
                save_agent_weight_list,
            ) = fafmodule.predict_all(data, 1, num_agent=num_agent)
        else:
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                data, 1, num_agent=num_agent
            )

        running_loss_class.update(cls_loss)
        running_loss_loc.update(loc_loss)
        # box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]
        box_color_map = ["red", "yellow", "blue", "purple", "black"]
        # If has RSU, do not count RSU's output into evaluation
        # eval_start_idx = 1 if args.rsu else 0
        eval_start_idx = 1 if args.rsu and "v2" in args.logpath else 0

        # local qualitative evaluation
        for k in range(eval_start_idx, num_agent):
            box_colors = None   # (N,) late fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                pred_restore = result[k][0][0][0]["pred"]   # current agent k [predictions_dicts=[[{}]],cls_pred_first_nms=tensor()]
                score_restore = result[k][0][0][0]["score"]
                selected_idx_restore = result[k][0][0][0]["selected_idx"]

            data_agents = {
                "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),          # torch.Size([5, 256, 256, 6, 6])
            }

            temp = gt_max_iou[k]

            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]
            # late fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                box_colors = late_fusion(
                    k, num_agent, result, trans_matrices, box_color_map
                )
                num_coper = len(np.unique(box_colors))
                box_color_map = box_color_map[:num_coper]


            result_temp = result[k] # result：each agent has a [predictions_dicts,cls_pred_first_nms]

            temp = {
                "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
                "result": [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou": data_agents["gt_max_iou"],
            }
            det_results_local[k], annotations_local[k] = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k])

            flag_coper = True
            bev_zeros = torch.zeros(temp["bev_seq"].shape)
            if np.array_equal(temp["bev_seq"],bev_zeros):
                flag_coper = False

            filename = str(filename0[0][0])
            cut = filename[filename.rfind("agent") + 7 :]
            seq_name = cut[: cut.rfind("_")]
            idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
            # logs/flag/rsu/test(val)/epoch_x/visk/scene_id
            seq_save = os.path.join(save_fig_path[k], seq_name)
            if args.visualization and flag_coper:
                check_folder(seq_save)
            idx_save = str(idx) + ".png"
            idx_save_grid = str(idx) + "_grid.png"
            temp_ = deepcopy(temp)
            # if args.visualization
            if args.visualization and flag_coper:
                visualization(
                    config,
                    temp,
                    box_colors,     # (N,) late fusion
                    box_color_map,  # box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"][:num_coper]
                    apply_late_fusion,
                    os.path.join(seq_save, idx_save),# logs/flag/rsu/test(val)/epoch_x/visk/scene_id/id.png
                    os.path.join(seq_save, idx_save_grid)
                )

            # plot the cell-wise edge
            if args.visualization:
            # if 1:
            #     check_folder(seq_save)
                if flag == "disco" and k < len(save_agent_weight_list):
                    # one_agent_edge: [w_0->k, w_1->k, ..., w_4->k]
                    one_agent_edge = save_agent_weight_list[k]
                    for kk in range(len(one_agent_edge)):
                        plt.figure()
                        idx_edge_save = (
                            str(idx) + "_edge_" + str(kk) + "_to_" + str(k) + ".png"
                        )
                        idx_edge_save_stick = (
                            str(idx) + "_edge_" + str(kk) + "_to_" + str(k) + "_grid" + ".png"
                        )
                        # logs/flag/rsu/test(val)/epoch_x/visk/scene_id/id_edge_kk_to_k.png
                        savename_edge = os.path.join(seq_save, idx_edge_save)
                        savename_edge_stick = os.path.join(seq_save, idx_edge_save_stick)

                        # flip
                        sns.heatmap(torch.flip(one_agent_edge[kk].cpu(),[0]),cmap='OrRd',cbar=True,vmin=0,vmax=1)
                        # sns.heatmap(one_agent_edge[kk].cpu(),cmap='OrRd',cbar=True,vmin=0,vmax=1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(savename_edge, dpi=500)

                        tick_positions = np.arange(0,33)
                        tick_labels = np.arange(0, 33)
                        plt.xticks(tick_positions,tick_labels,rotation=90)
                        plt.yticks(tick_positions,tick_labels[::-1],rotation=0)
                        plt.tick_params(axis='both', labelsize=8)
                        plt.grid(True, color='black', linewidth=0.5, linestyle='-', which='both', alpha=0.5)
                        plt.savefig(savename_edge_stick, dpi=500)

                        plt.close(1)


            # == tracking ==
            if args.tracking and flag_coper:
                scene, frame = filename.split("/")[-2].split("_")
                check_folder(tracking_path[k])
                det_file = os.path.join(tracking_path[k], f"det_{scene}.txt")
                if scene not in tracking_file[k]:
                    det_file = open(det_file, "w")
                    tracking_file[k].add(scene)
                else:
                    det_file = open(det_file, "a")
                det_corners = get_det_corners(config, temp_)
                for ic, c in enumerate(det_corners):
                    det_file.write(
                        ",".join(
                            [
                                str(
                                    int(frame) + 1
                                ),  # frame idx is 1-based for tracking
                                "-1",
                                "{:.2f}".format(c[0]),
                                "{:.2f}".format(c[1]),
                                "{:.2f}".format(c[2]),
                                "{:.2f}".format(c[3]),
                                str(result_temp[0][0][0]["score"][ic]),
                                "-1",
                                "-1",
                                "-1",
                            ]
                        )
                        + "\n"
                    )
                    det_file.flush()

                det_file.close()

            # restore data before late-fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                result[k][0][0][0]["pred"] = pred_restore
                result[k][0][0][0]["score"] = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_restore

        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Takes {} s\n".format(str(time.time() - t)))

    logger_root = args.logpath if args.logpath != "" else "logs"

    if args.data.endswith("val"):
        val_or_test = "_val"
    else:
        val_or_test = "_eval"

    if "disco+no_kd+MMI" not in args.resume:
        if "disco+no_kd" in args.resume and "MMI" not in args.resume: # disco+no_kd
            logger_root = os.path.join(logger_root, f"disco+no_kd"+val_or_test,"with_rsu" if args.rsu else "no_rsu")
        else:
            if args.MMI_flag:
                logger_root = os.path.join(logger_root, f"{flag}+MMI+" + str(args.alpha) +val_or_test,"with_rsu" if args.rsu else "no_rsu")
            else:
                logger_root = os.path.join(logger_root, f"{flag}"+val_or_test, "with_rsu" if args.rsu else "no_rsu")
    else: # disco+no_kd+MMI
        logger_root = os.path.join(logger_root, f"disco+no_kd+MMI+"+ str(args.alpha) +val_or_test,"with_rsu" if args.rsu else "no_rsu")

    print("logger_root = ", logger_root)
    os.makedirs(logger_root, exist_ok=True)
    if not apply_late_fusion:
        log_file_path = os.path.join(logger_root, "log_test_{}.txt".format(checkpoint["epoch"]))
    else:
        log_file_path = os.path.join(logger_root, "log_test_late{}.txt".format(checkpoint["epoch"]))

    log_file = open(log_file_path, "w")

    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    mean_ap_local = []
    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    for k in range(eval_start_idx, num_agent):
        if type(det_results_local[k]) != list or len(det_results_local[k]) == 0:
            continue
        print_and_write_log("\nLocal mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
            func_log = print_and_write_log
        )
        mean_ap_local.append(mean_ap)
        print_and_write_log("\nLocal mAP@0.7 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
            func_log = print_and_write_log
        )
        mean_ap_local.append(mean_ap)


        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    print_and_write_log("\n\nAverage local mAP@0.5")
    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
        func_log = print_and_write_log
    )
    mean_ap_local.append(mean_ap_local_average)

    print_and_write_log("\nAverage local mAP@0.7")
    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
        func_log = print_and_write_log
    )
    mean_ap_local.append(mean_ap_local_average)

    print_and_write_log(
        "\nQuantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, start_epoch - 1
        )
    )

    print("\nmean_ap_local = {}\n".format(mean_ap_local))
    for k in range(eval_start_idx, num_agent if "v2" in args.logpath else 4):
        print_and_write_log(
            "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                k + 1 if not args.rsu else k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
            )
        )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_local[-2], mean_ap_local[-1]
        )
    )

    print_and_write_log("\n{}\t{}\n".format(running_loss_class, running_loss_loc))

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", default=0, type=int, help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", type=int, default=0, help="Visualize validation result"
    )
    parser.add_argument(
        "--com",
        default="",
        type=str,
        help="lowerbound/upperbound/disco/when2com/v2v/sum/mean/max/cat/agent",
    )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument("--rsu", default=0, type=int, help="0: no RSU, 1: RSU")
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    parser.add_argument(
        "--MMI_flag",
        default=0,
        type=int,
        help="Whether to enable MMI",
    )

    parser.add_argument(
        "--alpha",
        default=0,
        type=float
    )
    parser.add_argument(
        "--flag_GPU",
        default=0,
        type=int
    )
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    # # # debug params
    # args.data =CMiMC/data/V2X-Sim-1-det/test'
    # args.com = 'lowerbound'
    # args.nepoch = 110
    # args.rsu = 1
    # args.MMI_flag = 0
    # args.lr_MMI = 0.0001
    # args.weight_LMI = 1
    # args.weight_GMI = 0.5
    # args.weight_miloss = 100
    # args.alpha = 0.4
    # args.log = False
    # args.seed = 1
    # args.visualization = 1
    # args.apply_late_fusion = 1
    # args.logpath = 'CMiMC/tools/det/logs_v1/'
    # args.resume = os.path.join(args.logpath,args.com,"with_rsu","epoch_{}.pth".format(args.nepoch))
    # args.logpath = 'CMiMC/tools/det/logs_v1_DIM_seed{}/logs_lr{}_{}_L{}+G{}'.format(args.seed,args.lr_MMI,args.weight_miloss,args.weight_LMI,args.weight_GMI)
    # args.resume = os.path.join(args.logpath,"xxxx+"+str(args.alpha),"with_rsu","epoch_{}.pth".format(args.nepoch))
    # # =============

    print(args)
    main(args)
