(
    mkdir /home/data_gen/3DAVS_benchmark/generation_pipeline/generated_dataset/part_1/5ZKStnWn8Zo/1/data_for_feature_gs &&
    cd /home/data_gen/3DAVS_benchmark/generation_pipeline/generated_dataset/part_1/5ZKStnWn8Zo/1/ &&
    cp -r distorted data_for_feature_gs &&
    cp -r images data_for_feature_gs &&
    cp -r visual_features_denseav data_for_feature_gs &&
    zip data_for_feature_gs.zip data_for_feature_gs
 )