function res_grappa = jh_ARC(kspace, AtA, kSize, lambda)

[feRes, peRes, cRes_vcc, dRes, eRes] = size(kspace);

cRes = cRes_vcc/2;
res_grappa = zeros(feRes, peRes, cRes, dRes*eRes);

parfor dd = 1:dRes*eRes
    for cc = 1:cRes
        res_grappa(:,:,cc,dd) = ARC(kspace, AtA, kSize, cc, dd, lambda);
    end
end



function [res_grappa] = ARC(kspace, AtA, kSize, cc, dd,lambda)
[feRes, peRes, cRes, dRes, eRes] = size(kspace);

s = [feRes+kSize(1)-1, peRes+kSize(2)-1, cRes, dRes, eRes];

kspace_zp = zeros(s);

kspace_zp(floor(s(1)/2)+1-ceil(feRes/2) : floor(s(1)/2)+ceil(feRes/2),...
          floor(s(2)/2)+1-ceil(peRes/2) : floor(s(2)/2)+ceil(peRes/2),:,:,:) = kspace;
      
tmp = zeros(kSize(1), kSize(2), cRes, dRes*eRes);
tmp((end+1)/2, (end+1)/2, cc, dd) = 1;
idxy = find(tmp);
clear tmp;

MaxListLen = 10000;
res_grappa = zeros(feRes,peRes);

LIST = zeros(kSize(1)*kSize(2)*cRes*dRes*eRes,MaxListLen);
KEY =  zeros(kSize(1)*kSize(2)*cRes*dRes*eRes,MaxListLen);

count = 0;
keys = 1;

for pp = 1:peRes
    nn = 1;
    for ff= 1:feRes
        tmp_ac_data = kspace_zp(ff:ff+kSize(1)-1,pp:pp+kSize(2)-1,:,:);
        pat = abs(tmp_ac_data)>0;

        if pat(idxy) | sum(pat)==0
            res_grappa(ff,pp) = tmp_ac_data(idxy);
        else
            key = pat(:);

%             if sum(key == KEY(:,nn)) == length(key)  % 같은 패턴이 있는경우
%                 idx = nn;
%             else  % 새로운 패턴이 들어온 경우
                idx = 0;

                for nn=1:keys 
                    if sum(key==KEY(:,nn))==length(key)
                       idx = nn;
                       break;
                    end
                end

%             end

            if idx == 0  % 새로운 패턴이 들어온 경우
                keys = keys + 1;
                count = count + 1;
                KEY(:,mod(count,MaxListLen)+1) = key(:);
                
                kernel = jh_Calibrate_d_util(AtA,kSize,cRes,dRes*eRes,cc,dd,lambda,pat);

                LIST(:,mod(count,MaxListLen)+1) = kernel(:);
            else  % 같은 패턴이 있는경우
                kernel = LIST(:,idx);
            end

            res_grappa(ff,pp) = sum(kernel(:).*tmp_ac_data(:));
        end
    end
end

% disp('ARC Finished');