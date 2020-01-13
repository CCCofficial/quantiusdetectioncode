function opts = getopts()

opts.batchSize = 100;
opts.numEpochs = 15;
pts.continue = true;
opts.gpus = [];
opts.learningRate = 10.^(-linspace(3,7,15)); %linspace(0.01, 0.0001, 15); %[repmat(0.01, 1, 5), repmat(0.001, 1, 5), repmat(0.0001, 1, 5)];


opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.weightDecay = 0.001; %0.0005 ;
opts.momentum = 0.9 ; % not valid

opts.saveMomentum = true ;
opts.nesterovUpdate = false ;

opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = {} ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;

% opts = vl_argparse(opts, varargin) ;