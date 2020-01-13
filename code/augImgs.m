function augSets = augImgs(im, factor)
switch factor
    case 4
        augSets = augmentby4(im);
    case 8
        error('Not supported');
    otherwise
        error('Unrecognized Input');
end
end

function augset = augmentby4(im)
nCh = size(im, 3);
augset = zeros(size(im, 1), size(im, 2), nCh, 4);
for n = 1 : nCh
    imSingleCh = im(:, :, n, 1);
    augset(:, :, n, 1) = imSingleCh;
    augset(:, :, n, 2) = fliplr(imSingleCh);
    augset(:, :, n, 3) = flipud(imSingleCh);
    augset(:, :, n, 4) = fliplr(flipud(imSingleCh));
end
end