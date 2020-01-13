function [prec, recall, fscore, objctv] = evalThisNet(model_dir, epoch, nset, ch_mean, ch_std, conf, flag)

% vggcellnet = loadvggcellnet(model_dir, epoch, nset, 'smallCNN');
load(fullfile(model_dir, sprintf('net-epoch-%d-fold-%d-%s.mat', epoch, nset, 'smallCNN')), 'info', 'vggcellnet');
augfull_DS = fullfile(conf.localdir, 'partitions', 'tstfull.mat');
load(augfull_DS, 'imdb');
imdb.images.data = nomalizeIMDB(imdb.images.data, ch_mean, ch_std);
vggcellnet.layers(end) = [] ;
res = vl_simplenn(vggcellnet, imdb.images.data) ;
preds = squeeze(res(end).x) ; %for eval call to vl_cnn the res(end-1).x is not reqd
N = size(preds, 3);

GTbyXperts = getAugGTbyXprts(conf); % M x N x 20 x 6
assert(N == size(GTbyXperts, 3), 'Wrong dimension');
Nxprt = size(GTbyXperts, 4);
tp = zeros(N, Nxprt); fp = zeros(N, Nxprt); fn = zeros(N, Nxprt);

for imid = 1 : N
    detxy = nmsSS(preds(:,:,imid), [25 25], 1.0); % T_n = 5 fornowl
    xdt = round((detxy(:, 2)+detxy(:, 4))/2);
    ydt = round((detxy(:, 3)+detxy(:, 5))/2);
    for nX = 1 : Nxprt
        [yGT, xGT] = find(squeeze(GTbyXperts(:,:,imid, nX)));     
        [~, ~, tp(imid, nX), fp(imid, nX), fn(imid, nX)] = ...
            evalDetect(xdt, ydt, xGT, yGT, ones(size(GTbyXperts, 1), size(GTbyXperts, 2)), 10);        
    end
    if flag,
        [ygt, xgt] = find(squeeze(imdb.images.dots(:,:,1,imid)));
        figure(28); subplot(1, 2, 1), imagesc(imdb.images.data(:,:,2,imid)); colormap gray; axis on; axis image; colorbar; hold on;
        plot(xgt, ygt, 'y*');
        plot(xdt, ydt, 'r*');
        hold off;
        figure(28); subplot(1, 2, 2), imagesc(preds(:,:,imid)); colormap gray; axis on; axis image; colorbar; hold on;
        plot(xgt, ygt, 'y*');
        plot(xdt, ydt, 'r*');
        hold off;
        pause;
    end
    
end

TP = sum(tp); FP = sum(fp); FN = sum(fn);
prec = TP./(TP + FP);
recall = TP./(TP + FN);
fscore = 2*(prec.*recall)./(prec+recall);
objctv = info.train.objective(end);
end

function data = nomalizeIMDB(data, ch_mean, ch_std)
nCh = size(data, 3);
for eachEchannel = 1 : nCh
    data(:,:,eachEchannel,:) = (data(:,:,eachEchannel,:) - ch_mean(eachEchannel))/ch_std(eachEchannel);
end
end