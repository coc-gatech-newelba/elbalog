# Automatic Anomaly Detection from System and Application Logs

## Datasets

### BG/L
System logs collected from supercomputer Blue Gene/L at Lawrence Livermore National Lab.
* 4,747,963 log entries
* 348,698 (7%) log entries labeled as anomalies
* 385 log message types
* Source: https://www.usenix.org/cfdr-data

### BG/P
System logs collected from supercomputer Blue Gene/P at Argonne National Laboratory.
* 2,084,392 log entries
* 138,172 (7%) log entries labeled as anomalies
* 219 log message types
* Source: https://www.usenix.org/cfdr-data

### HDFS
Application logs generated through running Hadoop MapReduce jobs on hundreds of nodes.
* 11,197,705 log entries
* 575,056 execution sessions
* 16,837 (3%) execution sessions labeled as anomalies
* 29 log message types
* Source: http://iiis.tsinghua.edu.cn/~weixu/sospdata.html

### CPU
CPU utilization logs generated through running the RUBBoS benchmark on Emulab PC3000 machines.

## Models

### Statistical Models
* Decision Tree
* Logistic Regression
* Support Vector Machine
* Principal Component Analysis

### Sequence-based Models
* Simple Recurrent Network (SRN)
* Gated Recurrent Unit (GRU) Network
* Long Short Term Memory (LSTM) Network
