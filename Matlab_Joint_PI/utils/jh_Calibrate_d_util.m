function kernel = jh_Calibrate_d_util(AtA, kSize, cRes, dRes, cc, dd, lambda, sampling)

dummyK = zeros(kSize(1),kSize(2),cRes,dRes);
dummyK((end+1)/2,(end+1)/2,cc,dd) = 1;
idxY = find(dummyK);
sampling(idxY) = 0;
idxA = find(sampling);

Aty = AtA(:,idxY);
Aty = Aty(idxA);


AtA = AtA(idxA,:);
AtA =  AtA(:,idxA);

kernel = sampling*0;

lambda = norm(AtA,'fro')/size(AtA,1)*lambda;

rawkernel = (AtA + eye(size(AtA))*lambda)\Aty;
kernel(idxA) = rawkernel; 



