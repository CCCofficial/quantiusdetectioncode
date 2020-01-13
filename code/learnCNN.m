function learnCNN(vggcellnet, opts, imdb, kfold, mthd)

vggcellnet.layers{end-1}.precious = 1;
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end
 
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    for i=1:numel(vggcellnet.layers)
        if isfield(vggcellnet.layers{i}, 'weights')
            J = numel(vggcellnet.layers{i}.weights) ;
            for j=1:J
                vggcellnet.layers{i}.momentum{j} = zeros(size(vggcellnet.layers{i}.weights{j}), 'single') ;
            end            
            if ~isfield(vggcellnet.layers{i}, 'learningRate')
                vggcellnet.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(vggcellnet.layers{i}, 'weightDecay')
                vggcellnet.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end


%% load from previous training
modelPath = @(ep, kfold, method) fullfile(opts.expDir, sprintf('net-epoch-%d-fold-%d-%s.mat', ep, kfold, method));

start = 0;

%% Launch training
fprintf('\n Entering Training: %d; Validation Set: %d \n', length(opts.train), length(opts.val));

for epoch=start+1:opts.numEpochs
    
    learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    train = opts.train(randperm(numel(opts.train))) ; % shuffle
    val = opts.val ;
    
    %
    [vggcellnet,stats.train] = process_epoch(opts, epoch, train, learningRate, imdb, vggcellnet) ;
    
    if evaluateMode, sets = {'val'} ; else sets = {'train'} ; end
    for f = sets
        f = char(f);
        n = numel(eval(f)) ; 
        info.(f).speed(epoch) = n / stats.(f)(1);
        info.(f).objective(epoch) = stats.(f)(2) / n ;
        info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
    end
    if ~evaluateMode
        fprintf('%s: saving model for epoch %d\n', mfilename, epoch) ;
        tic ;
        save(modelPath(epoch, kfold, mthd), 'vggcellnet', 'info') ;
        fprintf('%s: model saved in %.2g s\n', mfilename, toc) ;
    end
    
end
end

% -------------------------------------------------------------------------
function [net, stats] = process_epoch(opts, epoch, subset, learningRate, imdb, net)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% assume validation mode if the learning rate is zero
training = learningRate > 0 ;
if training
  mode = 'train' ;
  evalMode = 'normal' ;
else
  mode = 'val' ;
  evalMode = 'test' ;
end

res = [] ;
stats = [];
start = tic ;
error = [];
for t=1:opts.batchSize:numel(subset)
  fprintf('\n%s: epoch %02d: %3d/%3d:', mode, epoch, ...
          fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  numDone = 0;
  
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    [im, labels, dots] = getucsfBatch(imdb, batch) ; 

    if training
        dzdy = single(1); 
    else
        dzdy = [] ;
    end
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', evalMode, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn);
    
    % accumulate errors
    if 0,
        error = [res(end).x; 1-detbyreg(dots, res(end-1).x, im, labels, true)];
    else
        error = [res(end).x; 0];
    end
    numDone = numDone + numel(batch) ;
  end % next sub-batch

  % accumulate gradient
  if training
    [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
  end
end
time = toc(start) ;
stats(1) = time ;
stats(2) = error(1);
stats(3) = error(2); %/ceil(numel(subset)/opts.batchSize);

end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res)
% -------------------------------------------------------------------------

for l=numel(net.layers):-1:1
  for j=numel(res(l).dzdw):-1:1

      % Standard gradient training.
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;

      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
        thisLR * net.layers{l}.momentum{j} ;
    
  end
end
end

% --------------------------------------------------------------------
function [im, labels, dots] = getucsfBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch);
labels = imdb.images.labels(:,:,:,batch);
dots = imdb.images.dots(:,:,:,batch);
end

% for debugging and visualization only
function FScore = detbyreg(dots, pred, im, labels, flag)
% -------------------------------------------------------------------------
tol_radius = 5;
assert(ndims(dots) == 4, 'No of dimension should be 4 for dots in detbyreg()');
assert(ndims(pred) == 4, 'No of dimension should be 4 for pred in detbyreg()');
assert(ndims(im) == 4, 'No of dimension should be 4 for im in detbyreg()');
assert(ndims(labels) == 4, 'No of dimension should be 4 for labels in detbyreg()');
N = size(pred, 4);

dots = squeeze(dots); pred = squeeze(pred); labels = squeeze(labels);

tp = zeros(N,1); fp = zeros(N,1); fn = zeros(N,1);
for n = 1 : 20: N
    figure(15), subplot(1, 3, 1), imagesc(im(:,:,2,n)), colormap(gray), axis image; axis on, colorbar;
    figure(15), subplot(1, 3, 2), imagesc(pred(:,:,n)), colormap(gray), axis image; axis on, colorbar;
    figure(15), subplot(1, 3, 3), imagesc(labels(:,:,n)), colormap(gray), axis image; axis on, colorbar;
    
    figure(20), subplot(1, 3, 1), imagesc(im(:,:,1,n)), colormap(gray), axis image; axis on, colorbar;
    figure(20), subplot(1, 3, 2), imagesc(pred(:,:,n)), colormap(gray), axis image; axis on, colorbar;
    figure(20), subplot(1, 3, 3), imagesc(labels(:,:,n)), colormap(gray), axis image; axis on, colorbar;
    
    [yGt, xGt] = find(dots(:,:,n));
    detxy = nmsSS(pred(:,:,n), [10 10], 3.0); % T_n = 5 fornowl
    xdt = round((detxy(:, 2)+detxy(:, 4))/2);
    ydt = round((detxy(:, 3)+detxy(:, 5))/2);
    figure(25), subplot(1, 2, 1), imagesc(im(:,:,2,n)), colormap(gray), axis image, axis on; hold on;
    plot(xGt, yGt, 'g*'); hold off;
    figure(25), subplot(1, 2, 2), imagesc(pred(:,:,n)), colormap(gray), axis image; axis on; hold on;
    plot(xdt, ydt, 'r*'); plot(xGt, yGt, 'g*'); hold off;
    pause;
end
FScore = 0;

end
