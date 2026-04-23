i=input('Which case? (1,2,3) ');
if i==1 | i==2
  a=0; b=1;
elseif i==3
  a=-1; b=1;
end
n=input('How many subintervals? ');
[x,sol]=finel(i,a,b,n);
figure(1)
clf;
set(1,'defaultlinelinewidth',3)
set(1,'position',[407 12 612 684]);
fh=fplot(@(t)exact(i,t),[a b],'r');
fh.LineWidth = 3;
hold on;
plot(x,sol,'*--b')
lh=legend('exact','computed');
lh.FontSize = 20;
