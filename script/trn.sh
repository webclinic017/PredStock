#!/bin/sh
# apt update
# apt upgrade -y
rm /home/toshi/PROJECTS/PredStock/data_j.xls
wget https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls -P /home/toshi/PROJECTS/PredStock
chown -R toshi /home/toshi/PROJECTS/PredStock/data_j.xls

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

conda activate ag
chown -R toshi /home/toshi/PROJECTS/PredStock/AutogluonModels/*

# ipython /home/toshi/PROJECTS/PredStock/script/0a_HISTORY.py
ipython /home/toshi/PROJECTS/PredStock/script/1a_MOMENTUM.py &
ipython /home/toshi/PROJECTS/PredStock/script/1b_BETA.py &
wait
ipython /home/toshi/PROJECTS/PredStock/script/1c_XDAY.py
ipython /home/toshi/PROJECTS/PredStock/script/1d_CANNIK.py
ipython /home/toshi/PROJECTS/PredStock/script/1e_SCALE.py
ipython /home/toshi/PROJECTS/PredStock/script/1f_DIVSPLIT.py
ipython /home/toshi/PROJECTS/PredStock/script/1g_BETAJPY
ipython /home/toshi/PROJECTS/PredStock/script/1h_CANJPY.py

ipython /home/toshi/PROJECTS/PredStock/script/1x_FEATURE.py &
ipython /home/toshi/PROJECTS/PredStock/script/1y_LABEL.py &
wait
ipython /home/toshi/PROJECTS/PredStock/script/1z_FLJOIN.py
ipython /home/toshi/PROJECTS/PredStock/script/2a_TRAIN_OC3.py
ipython /home/toshi/PROJECTS/PredStock/script/3a_POST.py
ipython /home/toshi/PROJECTS/PredStock/script/4a_PREDICT.py

chown -R toshi /home/toshi/PROJECTS/PredStock/AutogluonModels/*
ipython /home/toshi/PROJECTS/PredStock/script/99_SHUTDOWN.py
