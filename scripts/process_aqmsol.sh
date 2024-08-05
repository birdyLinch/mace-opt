python -u mace/cli/preprocess_data.py \
    --train_file="/mnt/petrelfs/linchen/FoundationalModel/data/aqm/AQM-sol.hdf5" \
    --valid_fraction=0.05 \
    --r_max=5.0 \
    --h5_prefix="/mnt/petrelfs/linchen/FoundationalModel/data/aqm/processed_aqmsol/" \
    --compute_statistics \
    --E0s="{1: -13.64332105,6: -1027.61074626,7: -1484.27621709,8: -2039.75167568,9: -2710.54812971,15: -9283.01586200,16: -10828.72622208,17: -12516.46233936}" \
    --seed=123 \
    --h5_positions_key="atXYZ" \
    --h5_energy_key="ePBE0+MBD" \
    --h5_forces_key="totFOR" \
    --h5_numbers_key="atNUM" \

    
    
    
# no force key: filling 0.0s
# --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \ # default None