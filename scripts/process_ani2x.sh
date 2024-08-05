python -u mace/cli/preprocess_data.py \
    --train_file="/mnt/petrelfs/linchen/FoundationalModel/data/ani-2x/final_h5/ANI-2x-wB97X-631Gd.h5" \
    --valid_fraction=0.05 \
    --r_max=5.0 \
    --h5_prefix="/mnt/petrelfs/linchen/FoundationalModel/data/ani-2x/processed_wB97X-631Gd/" \
    --compute_statistics \
    --E0s="{35: -70045.28385080204, 6: -1030.5671648271828, 17: -12522.649269035726, 9: -2715.318528602957, 1: -13.571964772646918, 53: -8102.524593409054, 7: -1486.3750255780376, 8: -2043.933693071156, 15: -9287.407133426237, 16: -10834.4844708122}" \
    --seed=123 \
    --h5_positions_key="atXYZ" \
    --h5_energy_key="ePBE0+MBD" \
    --h5_forces_key="totFOR" \
    --h5_numbers_key="atNUM" \

# no force key: filling 0.0s
# --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \ # default None