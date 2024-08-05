python -u mace/cli/preprocess_data.py \
    --train_file="/mnt/petrelfs/linchen/FoundationalModel/data/aqm/AQM-gas.hdf5" \
    --valid_fraction=0.05 \
    --r_max=5.0 \
    --h5_prefix="/mnt/petrelfs/linchen/FoundationalModel/data/aqm/processed_aqmgas/" \
    --compute_statistics \
    --E0s="{1: -13.64140416,6: -1027.60791501,7: -1484.27481909,8: -2039.75030551,9: -2710.54734321,15: -9283.01120605,16: -10828.72289453,17: -12516.46004579}" \
    --seed=123 \
    --h5_positions_key="atXYZ" \
    --h5_energy_key="ePBE0+MBD" \
    --h5_forces_key="totFOR" \
    --h5_numbers_key="atNUM" \

    
    
    
# no force key: filling 0.0s
# --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \ # default None