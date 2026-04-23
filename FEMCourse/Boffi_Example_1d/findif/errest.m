clear all
%
% error estimate using "findif" (vector norm)
%
test=2;
n=[2 4 8 16 32 64 128 256 512];
for i=1:length(n)
  [x sol]=findif(test,0,1,n(i));
  normer(i)=norm(exact(test,x)-sol,'inf')/norm(exact(test,x),'inf');
end
for i=1:length(n)-1
  order(i)=-(log(normer(i+1)/normer(i))/log(n(i+1)/n(i)));
end
order
figure(1)
clf;
set(1,'defaultlinelinewidth',3)
set(1,'position',[407 12 612 684]);
loglog(n,normer,'b')
