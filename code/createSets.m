function createSets(im_all, gt_all, method, opt_meanstd, setId, savefilename, force)

if exist(savefilename, 'file') && ~force
    return;
end

switch method
    case 'augtiles'        
        imdb = getDatafromAugTiles(im_all, gt_all, setId);
    case 'augfull'        
        imdb = getDatafromAugFull(im_all, gt_all, setId);
    case 'onlyfull'        
        imdb = getDatafromFullImg(im_all, gt_all, setId);
    otherwise
        error('unrecognized input inside createSetsVGGStyle()');
end

if opt_meanstd,
    [ch_mean, ch_std] = computemean(imdb.images.data);
    imdb.ch_mean = ch_mean;
    imdb.ch_std = ch_std;
end

save(savefilename, 'imdb');
end

function imdb = getDatafromAugTiles(im_all, gt_all, setId)
R = size(gt_all, 1) ; C = size(gt_all, 2);
tilesz = [100 100];
shift = tilesz/2;
count = 0;
for nthImg = 1 : size(gt_all, 4) 
    % data augmentation by tiling
    for row = 1 : shift(1) : R - tilesz(1) + 1 % 100 x 100 tiles
        for col = 1 : shift(2) : C - tilesz(2) + 1 % 100 x 100 tiles
            imchnnl = im_all(row:row + tilesz(1) - 1, col:col + tilesz(2) - 1, :, nthImg);
            gtdata = gt_all(row:row + tilesz(1) - 1, col:col + tilesz(2) - 1, 1, nthImg);
            labels = adjustTargetVGG(gtdata);
            
            if (sum(gtdata(:)) == 0), 
                continue;  % just to avoid plain background
            else
            
                bfs = augImgs(imchnnl, 4);
                gts = augImgs(gtdata, 4);
                lbls = augImgs(labels, 4); 

                for numaug = 1 : size(gts, 4)
                    count = count + 1;
                    imdb.images.data(:,:,:,count) = single(bfs(:,:,:,numaug));
                    imdb.images.dots(:,:,1,count) = single(gts(:,:,numaug));
                    imdb.images.labels(:,:,1,count) = single(lbls(:,:,numaug));
                    imdb.images.id(count, :) = [nthImg, col, row];
                    imdb.images.set(count, 1) = setId;
                end
                clear bfs grns gts lbls;                             
                clear bfchnl grnchnl gtdata labels;
            end
            
        end % for loop for col
    end % for loop for row
end
end

function imdb = getDatafromAugFull(im_all, gt_all, setId)
% data augmentation from full image
count = 0;
for nthImg = 1 : size(gt_all, 4)
    chn = augImgs(im_all(:,:,:,nthImg), 4); 
    gts = augImgs(gt_all(:,:,1,nthImg), 4);
    lbls = augImgs(adjustTargetVGG(gt_all(:,:,1,nthImg)), 4);
    
    for numaug = 1 : size(gts, 4)
        count = count + 1;
        imdb.images.data(:,:,:,count) = single(chn(:,:,:,numaug));
        imdb.images.dots(:,:,1,count) = single(gts(:,:,1,numaug));
        imdb.images.labels(:,:,1,count) = single(lbls(:,:,1,numaug));
        imdb.images.id(count, :) = [nthImg, 1, 1];
        imdb.images.set(count, 1) = setId;
    end
    clear bfs grns gts lbls;
end
end

function imdb = getDatafromFullImg(im_all, gt_all, setId)
% No data augmentation: just saving the full image
for nthImg = 1 : size(gt_all, 4) 
    imdb.images.data(:,:,:,nthImg) = single(im_all(:,:,:,nthImg));
    imdb.images.dots(:,:,1,nthImg) = single(gt_all(:,:,1,nthImg));
    imdb.images.labels(:,:,1,nthImg) = single(adjustTargetVGG(gt_all(:,:,1,nthImg)));
    imdb.images.id(nthImg, :) = [nthImg, 1, 1];
    imdb.images.set(nthImg, 1) = setId;
end
end


function [chn_mean, chn_std] = computemean(im_all)
nCh = size(im_all, 3);
chn_mean = zeros(nCh, 1, 'single');
chn_std = zeros(nCh, 1, 'single');
for eachChn = 1 : nCh
    chnnls = im_all(:,:,eachChn,:);
    chn_mean(eachChn) = mean(chnnls(:));
    chn_std(eachChn) = std(chnnls(:));
end
end

function target = adjustTargetVGG(pos)
pos = squeeze(pos);
% Create pixel-level labels to compute the loss
kernelsz = 5; hfkernelsz = (kernelsz-1)/2;
[mx, my] = meshgrid(-hfkernelsz:hfkernelsz, -hfkernelsz:hfkernelsz);
gaussianpeak = 7*exp(-(mx.*mx+my.*my)/8);
target = zeros(size(pos)+2*hfkernelsz,'single') ;
[y, x] = find(pos);
for nt = 1 : length(y)
    target(y(nt):y(nt)+kernelsz-1,x(nt):x(nt)+kernelsz-1) = gaussianpeak;
end
target = target(2:size(pos, 1)+1, 2:size(pos, 2)+1);
end
