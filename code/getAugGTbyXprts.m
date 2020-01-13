function GTbyXperts = getAugGTbyXprts(conf)

M = 500; N = 500; 
XpertNames = {'AJH', 'AKP', 'CK', 'DP', 'HM', 'RW'};
filename = @(xpert_name) (sprintf(conf.Annotation.allexperts4test, xpert_name));
% get annotation
nImg = 5; 
GTbyXperts = zeros(M, N, nImg, length(XpertNames), 'single');
for XpertID = 1 : 6
    count = 0;
    for nthim = 1 : 5
        annofile = filename(XpertNames{XpertID});
        annotations = csvread(annofile);       
        xGT = min(max(round(annotations(annotations(:, 3) == nthim, 1)), 1), N);
        yGT = min(max(round(annotations(annotations(:, 3) == nthim, 2)), 1), M);
        gt = zeros(M, N, 1);
        gt(yGT+(xGT-1)*M) = 1;
        auggts = augImgs(gt, 4);
        for nrot = 1 : size(auggts, 4)
            count = count + 1;
            GTbyXperts(:,:,count, XpertID) = auggts(:,:,nrot);
        end
        clear xGT yGT;
    end    
end
end

