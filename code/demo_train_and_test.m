%% get dataset dircetory
clc, clear all, close all;

%% arrange file names for training and testing
allImgs = loaducsfMetaData(dbinfo());
makePartitions(allImgs, dbinfo());

%% load images and corresponding ground truth for training and testing
loadImgGT(dbinfo());

%% generate partitions
genTrainTestData(dbinfo());

%% Pretraining
% At this point you should add MatConvNet (http://www.vlfeat.org/matconvnet/) to your path
run('../dependency/matcnn/cvPracticalsReg/practical-cnn-reg-2016a/setup.m');
pretrain(dbinfo());

%% Finetuning
finetune(dbinfo());

%% Evaluation: final performance
evaluateOnTestSet(dbinfo());