function evaluateOnTestSet(conf)

% data normalization parameters
eachSet = 1;
onlyfull_DS = fullfile(conf.localdir, 'partitions', ['trnfull_stage_' num2str(eachSet, '%0.2d'), '.mat']);
% fullfile(conf.localdir, 'trnfull.mat');
load(onlyfull_DS, 'imdb');
ch_mean = imdb.ch_mean; ch_std = imdb.ch_std;

% location of learnt model
finetune_model_dir = fullfile(conf.localdir, 'partitions');

for n = 1:15 % learning iterations
    [prec, recall, fscore, objctv] = evalThisNet(finetune_model_dir, n, 0, ch_mean, ch_std, dbinfo, false);
    fprintf('\n iter %d: prec, %f; recall, %f; fscore, %f; objctv, %f', n, mean(prec), mean(recall), mean(fscore), objctv);
end