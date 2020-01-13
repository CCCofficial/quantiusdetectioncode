function pretrain(conf)
methodology = 'smallCNN';
augtile_DS = fullfile(conf.localdir, 'partitions', 'augtiles.mat');
load(augtile_DS, 'imdb');

pretrn_model_dir = fullfile(conf.localdir, 'pretrain');
if ~exist(pretrn_model_dir, 'dir'), mkdir(pretrn_model_dir); end;
% trainCNN(imdb, iFolds, pretrn_model_dir, [], methodology);
trainCNN(imdb, 0, pretrn_model_dir, [], methodology);