# Path to the original V2X-Sim dataset
original_data_path := ../data/V2X-Sim-1-origin
# Where to save the created data
create_data_save_path := ../data/V2X-Sim-1-det
# Index of the begining scene
scene_begin := 0
# Index of the ending scene + 1
scene_end := 100
# Index of the start agent
from_agent := 0
# Index of the end agent + 1
to_agent := 6
# [v2.0 / v2.0-mini]
dataset_version := v2.0

training_script := train_codet.py
# Path to the created training data
# training_data := $(create_data_save_path)/train
training_data := $(create_data_save_path)/train

testing_script := test_codet.py
# Path to the test/val data
testing_data := $(create_data_save_path)/test
# [lowerbound / upperbound / v2v / disco / when2com ]
com := disco
batch_size := 4

flag_GPU := 0

MMI_flag := 1
lr_MMI := 0.0001

nepoch := 130
weight_LMI := 0
weight_GMI := 1
alpha = 0.2
weight_miloss := 100

seed := 1
logpath := logs_seed$(seed)/logs_lr$(lr_MMI)_$(weight_miloss)_L$(weight_LMI)+G$(weight_GMI)

# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)
# compress_level 2^compress_level
compress_level := 0
# 1: apply late fusion. 0: no late fusion
apply_late_fusion := 0
# 1: do visualizaton. 0: no visualization
visualization := 0
# pose_noise in meters
pose_noise := 0
# only apply v2i communication
only_v2i := 0
# 0: no RSU, 1: RSU
rsu := 1
# Whether to use pose info for When2com
warp_flag := 0
# Used when testing when2com / who2com
# when2com: activated, who2com: argmax_test
inference := activated

create_data:
	python create_data_det.py \
	--root $(original_data_path) \
	--scene_begin $(scene_begin) \
	--scene_end $(scene_end) \
	--savepath $(create_data_save_path) \
	--from_agent $(from_agent) \
	--to_agent $(to_agent) \
	--dataset_version $(dataset_version)

train:
	python $(training_script) \
	--data $(training_data) \
	--com $(com) \
	--log \
	--batch_size $(batch_size) \
	--auto_resume_path $(auto_resume_path) \
	--nepoch $(nepoch) \
	--logpath $(logpath) \
	--warp_flag $(warp_flag) \
	--rsu $(rsu) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--flag_GPU $(flag_GPU)

train_disco:
	python $(training_script) \
	--data $(training_data) \
	--com disco \
	--log --batch $(batch_size) \
	--kd_flag 1 \
	--resume_teacher $(logpath)/upperbound/with_rsu/epoch_x.pth \
	--auto_resume_path $(auto_resume_path) \
	--logpath $(logpath) \
	--nepoch $(nepoch) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--rsu 1 \
	--flag_GPU $(flag_GPU)

train_disco_no_kd:
	python $(training_script) \
	--data $(training_data) \
	--com disco \
	--log --batch $(batch_size) \
	--kd_flag 0 \
	--auto_resume_path $(auto_resume_path) \
	--logpath $(logpath) \
	--nepoch $(nepoch) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--rsu 1 \
	--MMI_flag $(MMI_flag) \
	--alpha $(alpha) \
	--flag_GPU $(flag_GPU) \
	--weight_LMI $(weight_LMI) \
	--weight_GMI $(weight_GMI) \
	--weight_miloss $(weight_miloss) \
	--seed $(seed) \
	--lr_MMI $(lr_MMI)

train_disco_no_rsu:
	python $(training_script) \
	--data $(training_data) \
	--com disco \
	--log --batch $(batch_size) \
	--kd_flag 1 \
	--resume_teacher $(logpath)/upperbound/no_rsu/epoch_$(nepoch).pth \
	--auto_resume_path $(auto_resume_path) \
	--logpath $(logpath) \
	--nepoch $(nepoch) \
	--compress_level $(compress_level) \
	--pose_noise $(pose_noise) \
	--only_v2i $(only_v2i) \
	--rsu 0 \
	--flag_GPU $(flag_GPU)

test:
	python $(testing_script) \
	--data $(testing_data) \
	--com $(com) \
	--resume $(logpath)/$(com)/with_rsu/epoch_$(nepoch).pth \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1 \
	--flag_GPU $(flag_GPU)

test_disco_with_MMI:
	python $(testing_script) \
	--data $(testing_data) \
	--com disco \
	--resume $(logpath)/disco+MMI+$(alpha)/with_rsu/epoch_$(nepoch).pth \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1 \
	--MMI_flag 1 \
	--alpha $(alpha) \
	--flag_GPU $(flag_GPU)

test_disco_no_kd_no_MMI:
	python $(testing_script) \
	--data $(testing_data) \
	--com disco \
	--resume $(logpath)/disco+no_kd/with_rsu/epoch_$(nepoch).pth \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1 \
	--MMI_flag 0 \
	--flag_GPU $(flag_GPU)

test_disco_no_kd_with_MMI:
	python $(testing_script) \
	--data $(testing_data) \
	--com disco \
	--resume $(logpath)/disco+no_kd+MMI+$(alpha)/with_rsu/epoch_$(nepoch).pth \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1 \
	--MMI_flag 1 \
	--alpha $(alpha) \
	--flag_GPU $(flag_GPU)

test_no_rsu:
	python $(testing_script) \
	--data $(testing_data) \
	--com $(com) \
	--resume $(logpath)/$(com)/with_rsu/epoch_$(nepoch).pth \
	--tracking \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 0 \
	--flag_GPU $(flag_GPU)

test_warp:
	python $(testing_script) \
	--data $(testing_data) \
	--com when2com \
	--resume $(logpath)/when2com_warp/with_rsu/epoch_$(nepoch).pth \
	--tracking \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 1 \
	--inference $(inference) \
	--warp_flag 1 \
	--flag_GPU $(flag_GPU)

test_warp_no_rsu:
	python $(testing_script) \
	--data $(testing_data) \
	--com when2com \
	--resume $(logpath)/when2com_warp/no_rsu/epoch_$(nepoch).pth \
	--tracking \
	--logpath $(logpath) \
	--apply_late_fusion $(apply_late_fusion) \
	--visualization $(visualization) \
	--rsu 0 \
	--inference $(inference) \
	--warp_flag 1 \
	--flag_GPU $(flag_GPU)
