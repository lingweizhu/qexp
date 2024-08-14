# q-exponential family for policy optimization
This is the code base accompanying our paper "q-exponential family for policy optimization".

We included heavy-tailed and light-tailed distributions.



# online results 

All the run statistics are logged to a MySQL database server. The schema can be found in [`configs/schema/default-schema.yaml`](configs/schema/default-schema.yaml)
## How to run
0. This codebase contains some features that are only available in Python3.10+

1. Install requirements:
``` bash
pip install -r requirments.txt
``` 

2. Add the correct db credentials to [`configs/db/credentials.yaml`](configs/db/credentials.yaml). You can log the results to the local database or to a google cloud server (recommended). I would recommend making `credentials-local.yaml` and adding it to `.gitignore`.

3. Set `db_prefix` to your username in [`configs/config.yaml`](configs/config.yaml). This is especially required if hosting on CC because they only allow you to make databases that start with your CC username. 

4. Run the following command for a sweep of 2400 runs as defined in
   [`configs/config.yaml`](configs/config.yaml). The `args.run` argument
is the run id for the first experiment in the sweep. Other experiments
are automatically assigned a run id of args.run + sweep_id in the
database.

``` bash
python main.py run=0
```


# offline results 

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
```%
