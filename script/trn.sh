#!/bin/sh

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/toshi/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/toshi/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/toshi/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/toshi/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

chown -R toshi /home/toshi/PROJECTS/PredStock/AutogluonModels/*
conda activate ag2
ipython /home/toshi/PROJECTS/PredStock/script/06_PRE.py
ipython /home/toshi/PROJECTS/PredStock/script/07_TRAIN.py
ipython /home/toshi/PROJECTS/PredStock/script/08_POST.py
ipython /home/toshi/PROJECTS/PredStock/script/07_TRAIN_REG.py
ipython /home/toshi/PROJECTS/PredStock/script/08_POST.py
chown -R toshi /home/toshi/PROJECTS/PredStock/AutogluonModels/*
ipython /home/toshi/PROJECTS/PredStock/script/06_PREDICT.py
# rsync /home/toshi/PROJECTS/PredStock/ /mnt/qnap/home/PROJECTS/PredStock/ -avh --delete
ipython /home/toshi/PROJECTS/PredStock/script/99_SHUTDOWN.py
