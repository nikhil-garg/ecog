A=permute(X, [3,1,2]);
S=size(A);
train=reshape(A,[S(1)*S(2),S(3)]);
writematrix(train,"test_data.csv");

