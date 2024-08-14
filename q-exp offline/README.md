### Code submission for offline learning experiments

Generate scripts:

```
cd configs/
python config_v0.py
chmod +x scripts/tasks_*
cd ..
```

Run the script from the root.

Test one run:
```
./configs/scripts/tasks_0.sh
```

All runs:
```
parallel ./configs/scripts/tasks_{}.sh ::: $(seq 0 6224)
```