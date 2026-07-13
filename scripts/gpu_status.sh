#!/bin/bash

nvidia-smi \
--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
--format=csv,noheader
