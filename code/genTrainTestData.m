function genTrainTestData(conf)

%% Arrange data for pretraining: augment the tiles cropped from full image
load(fullfile(conf.localdir, 'partitions', ['setsImgs' num2str(1, '%0.2d') '.mat']), 'trnIms_all', 'trnGts_all');
% pretrainig
createSets(trnIms_all, trnGts_all, 'augtiles', true, 1, fullfile(conf.localdir, 'partitions', 'augtiles.mat'), true);

%% Arrange data for finetuning: full images for finetuning

load(fullfile(conf.localdir, conf.train_and_test_sets), 'trainSet');
nSets = length(trainSet);
for eachSet = 1:nSets
    % Training and Test indices
    load(fullfile(conf.localdir, 'partitions', ['setsImgs' num2str(eachSet, '%0.2d') '.mat']), 'trnIms_all', 'trnGts_all');
    % finetuning
    createSets(trnIms_all, trnGts_all, 'onlyfull', true, 1, fullfile(conf.localdir, 'partitions', ['trnfull_stage_' num2str(eachSet, '%0.2d'), '.mat']), true);
end

%% Arrange data for testing:: full images for testing
load(fullfile(conf.localdir, 'partitions', ['setsImgs' num2str(1, '%0.2d') '.mat']), 'tstIms_all', 'tstGts_all');
createSets(tstIms_all, tstGts_all, 'augfull', false, 1, fullfile(conf.localdir, 'partitions', 'tstfull.mat'), true);


