function rect = nmsSS(RM,szQ,T_n)
% non-maximum suppression at single scale
[M, N] = size(RM);
height = szQ(1); width = szQ(2); % col

half_h = fix(height/2); half_w = fix(width/2);

bbox = zeros(ceil(M/height)* ceil(N/width), 5); % max possible #detections

cnt = 0;
    
largeneg = T_n - 1000.0; % a large negative value

topscore = max(RM(:));

while topscore > T_n

    [r_ind,c_ind] = find(RM==topscore);    
    if numel(r_ind) > 1
        r_ind = r_ind(1);
        c_ind = c_ind(1);
    end
    
    r_b = max(1, r_ind-half_h);
    c_b = max(1, c_ind-half_w);
    r_e = min(M, r_b + height - 1);
    c_e = min(N, c_b + width - 1);

    if cnt > 0
        ovmax=-inf;
        for j=1:cnt
            bbgt=bbox(j, 2:end);
            bi=[max(c_b,bbgt(1)) ; max(r_b,bbgt(2)) ; min(c_e,bbgt(3)) ; min(r_e,bbgt(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 && ih>0
                % compute overlap as area of intersection / area of union
                ua=(c_e-c_b+1)*(r_e-r_b+1)+...
                    (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                    iw*ih;
                ov=iw*ih/ua;
                if ov>ovmax
                    ovmax=ov;
%                     jmax=j;
                end
            end
        end
    
        if ovmax < 0.4
            cnt = cnt + 1;
            bbox(cnt, :) = [topscore c_b r_b c_e r_e]; % multiple detection
            RM(r_b:r_e, c_b:c_e) = largeneg;             
        else
            RM(r_b:r_e, c_b:c_e) = largeneg; % non maximum suppression            
        end        
    else
        cnt = cnt + 1;
        bbox(cnt, :) = [topscore c_b r_b c_e r_e]; % first detection
        RM(r_b:r_e, c_b:c_e) = largeneg;        
    end   
    
    topscore = max(RM(:));
end

rect = bbox(1:cnt, :);

