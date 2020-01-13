function loadImgGT(conf)

load(fullfile(conf.localdir, conf.train_and_test_sets), 'trainSet');
nSets = length(trainSet);

for n = 1:nSets
    load(fullfile(conf.localdir, 'partitions', ['setsImgIds' num2str(n, '%0.2d') '.mat']), 'trainIdx', 'trainImgs', 'testIdx', 'testImgs');
    
    % Trn image loading
    trnIms_all = loadTrnImgs(conf, trainImgs);
    % ground truth loading
    trnGts_all = loadAlexGt(conf, trainIdx);
    
    % Tst image loading
    tstIms_all = loadTstImgs(conf, testImgs);
    % ground truth loading
    tstGts_all = loadAlexGt(conf, testIdx);
    
    save(fullfile(conf.localdir, 'partitions', ['setsImgs' num2str(n, '%0.2d') '.mat']), 'trnIms_all', 'trnGts_all', 'tstIms_all', 'tstGts_all');
    
end

end

function im_all = loadTrnImgs(conf, trainImgs)
totalTrImgs = length(trainImgs);
im_all = zeros(conf.image.height, conf.image.width, conf.image.nCh, totalTrImgs, 'single');
img1 = imread(trainImgs(1).data);
img1_topband = img1(56:105, :, 1:conf.image.nCh);

for num = 1 : totalTrImgs
    im_each = imread(trainImgs(num).data);
    im_each_2 = im_each(:,:,1:conf.image.nCh);
    im_each_2(1:50, :, :) = img1_topband;
    im_all(:, :, :, num) = single(im_each_2);
end
end

function gt_all = loadAlexGt(conf, all_im_ids)
Afull = csvread(sprintf(conf.Annotation.filename, num2str(conf.folderindx))); 
Aexpert = Afull(conf.TrAnnotation.rows.fst:conf.TrAnnotation.rows.lst,conf.TrAnnotation.cols.fst:conf.TrAnnotation.cols.lst); 
M = conf.image.height; N = conf.image.width;
gt_all = zeros(M, N, 1, length(all_im_ids), 'single');
for n = 1 : length(all_im_ids)
    imid = all_im_ids(n, 1);
    x = round(Aexpert(Aexpert(:, 1) == imid, 2)); x = min(max(x, 1), N);
    y = round(Aexpert(Aexpert(:, 1) == imid, 3)); y = min(max(y, 1), M);
    gt = zeros(M, N, 1, 'single');
    gt(y+(x-1)*M) = 1;
    gt_all(:,:,1,n) = gt;
end
end

function im_all = loadTstImgs(conf, trainImgs)
totalTrImgs = length(trainImgs);
im_all = zeros(conf.image.height, conf.image.width, conf.image.nCh, totalTrImgs, 'single');

for num = 1 : totalTrImgs
    im_each = imread(trainImgs(num).data);
    im_each_2 = im_each(:,:,1:conf.image.nCh);
    im_all(:, :, :, num) = single(im_each_2);
end
end
