function finetune(conf)
pretrn_model_dir = fullfile(conf.localdir, 'pretrain');


eachSet = 1;
onlyfull_DS = fullfile(conf.localdir, 'partitions', ['trnfull_stage_' num2str(eachSet, '%0.2d'), '.mat']);

load(onlyfull_DS, 'imdb');

finetune_model_dir = fullfile(conf.localdir, 'partitions');
if ~exist(finetune_model_dir, 'dir'), mkdir(finetune_model_dir); end;
finetuneCNN(imdb, finetune_model_dir, pretrn_model_dir, 'smallCNN');
