function conf = dbinfo()

conf.datadir = '../data';
conf.expdir = '../alex_release';
conf.localdir = './local';
conf.fileprefix = '%s_merged';
conf.train_and_test_sets = 'train_and_test_sets.mat';
conf.folderindx = 9;
conf.Annotation.filename = '../annotation/%s_gt_expert.csv';
conf.Annotation.allexperts4test = '../annotation/Results%s_edit_skb.csv';
conf.TrAnnotation.rows.fst = 1;
conf.TrAnnotation.rows.lst = 4396;
conf.TrAnnotation.cols.fst = 1;
conf.TrAnnotation.cols.lst = 3;
conf.image.height = 500; 
conf.image.width = 500;
conf.image.nCh = 2;
conf.bittype = 8; % 8-bit images

conf.nFolds = 1; % minimum two
conf.N_DSz = 48;

% following is the database structure
% imdb.images.data(:,:,:,count) -> 100 x 100 x 2 x N
% imdb.images.labels(:,:,1,count) -> 100 x 100 x 1 x N
% imdb.images.dots(:,:,1,count) -> 100 x 100 x 1 x N
% imdb.images.id(count, :) -> x y
% imdb.images.set(count) -> 'pretrain': 1, 'val': 2, 'finetune': 3, 'val2': 4, 'test': 5
% imdb.means.bf -> double
% imdb.means.green -> double