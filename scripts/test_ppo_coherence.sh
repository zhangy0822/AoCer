vg_img_feature_npy=YOUR_PROJECT_PATH/data/caches/vg_test_img_feat.npy
vg_img_logits_npy=YOUR_PROJECT_PATH/data/caches/vg_test_img_logits.npy
vg_img_property_npy=YOUR_PROJECT_PATH/data/caches/vg_test_img_property_logits.npy
txt_img_matching_pretrained_ckpt=YOUR_PROJECT_PATH/logs/text_image_matching_0308_102515/snapshots/49111.pkl
img_attr_pretrained_ckpt=YOUR_PROJECT_PATH/logs/image_attribute_0108_153147/snapshots/49.pkl
ppo_pretrained_ckpt=YOUR_PROJECT_PATH/logs/ppo_coherence_gcn_0609_190506/snapshots/500.pkl
pretrained_ckpt=YOUR_PROJECT_PATH/logs/ppo_coherence_gcn_0609_190506/snapshots/500.pkl

python test/test_ppo_coherence_copy.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --n_epochs=50 \
       --instance_dim=100 \
       --max_turns=10 \
       --test_turns=1 \
       --lr=1e-5 \
       --ppo_target_kl=0.7 \
       --ppo_num_actions=3 \
       --exp_name=ppo_coherence \
       --vg_img_feature=$vg_img_feature_npy \
       --vg_img_logits=$vg_img_logits_npy \
       --vg_img_property_logits=$vg_img_property_npy \
       --txt_img_matching_pretrained=$txt_img_matching_pretrained_ckpt \
       --img_attr_pretrained=$img_attr_pretrained_ckpt \
       --ppo_pretrained=$ppo_pretrained_ckpt \
       --pretrained=$pretrained_ckpt
