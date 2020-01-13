function makePartitions(allImgs, conf)

load(fullfile(conf.localdir, conf.train_and_test_sets), 'trainSet', 'testSet');

for n = 1:length(trainSet)
    % Training and Test indices
    testIdx = testSet{n}; % image indices for training set
    trainIdx = trainSet{n}; % image indices for test set
    trainImgs = struct('data', {allImgs(trainIdx).data});
    testImgs = struct('data', {allImgs(testIdx).data});
    % Train and Test dataset: image indices only
    save(fullfile(conf.localdir, 'partitions', ['setsImgIds' num2str(n, '%0.2d') '.mat']), 'trainIdx', 'trainImgs', 'testIdx', 'testImgs');
end