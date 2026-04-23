function [x,sol]=finel(i,a,b,n);
%
% [x,sol]=finel(i,a,b,n)
% solves -u''=f, u(a)=exact(a), u(b)=exact(b)
% with finite elements
% n=number of subintervals
% i selects the rhs
% needs: "f" and "exact"
%
h=(b-a)/n;
A=(diag(2*ones(1,n-1))-diag(ones(1,n-2),-1)-diag(ones(1,n-2),1))/h;
x=(a:h:b)';
%
% 3 nodes Gauss formula
%
xx=[-0.7745966692414833770359 0 0.7745966692414833770359];
bb1=xx/2+1/2;
bb2=-xx/2+1/2;
pp=[0.5555555555555555555556 0.888888888888888888889 ...
    0.555555555555555555556];
pp=pp/sum(pp);
g=zeros(n-1,1);
for ii=1:n-1
  xx1=bb1*h+x(ii);
  xx2=xx1+h;
  g(ii)=h*(sum(pp.*f(i,xx1).*bb1)+sum(pp.*f(i,xx2).*bb2));
end
%
sol=zeros(n+1,1);
sol(1)=exact(i,x(1)); sol(n+1)=exact(i,x(n+1));
g(1)=g(1)+sol(1)/h; g(n-1)=g(n-1)+sol(n+1)/h;
sol(2:n)=A\g;
