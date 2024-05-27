vg_img_feature_npy=YOUR_PROJECT_PATH/data/caches/vg_test_img_feat_global.npy
vg_img_logits_npy=YOUR_PROJECT_PATH/data/caches/vg_test_img_logits.npy
txt_img_matching_pretrained_ckpt=YOUR_PROJECT_PATH/logs/text_image_matching_0308_102515/snapshots/49111.pkl
img_attr_pretrained_ckpt=YOUR_PROJECT_PATH/logs/image_attribute_0108_153147/snapshots/49.pkl
vg_attr_feature_npy=YOUR_PROJECT_PATH/data/caches/vg_attribute_feat.npy
vg_obj_feature_npy=YOUR_PROJECT_PATH/data/caches/vg_object_feat.npy

python test/test_ppo_rules.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --n_epochs=50 \
       --instance_dim=100 \
       --max_turns=10 \
       --test_turns=1 \
       --lr=1e-5 \
       --ppo_target_kl=0.7 \
       --ppo_num_actions=10 \
       --ppo_rule=coherence \
       --exp_name=ppo_rule \
       --vg_img_feature=$vg_img_feature_npy \
       --vg_img_logits=$vg_img_logits_npy \
       --txt_img_matching_global_pretrained=$txt_img_matching_pretrained_ckpt \
       --img_attr_pretrained=$img_attr_pretrained_ckpt \
       --ppo_pretrained=$ppo_pretrained_ckpt \
       --pretrained=$pretrained_ckpt \
       --vg_attr_feat=$vg_attr_feature_npy \
       --vg_obj_feat=$vg_obj_feature_npy 
