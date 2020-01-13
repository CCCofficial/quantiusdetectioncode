function allImgs = loaducsfMetaData(conf)

prefix = sprintf(fullfile(conf.datadir, conf.fileprefix), num2str(conf.folderindx));

for n = 1 : conf.N_DSz    
    allImgs(n).data = [prefix, [num2str(n, '%0.2d') '.png']];
end

if 0 % display all names
     [{allImgs(:).data}]'
end