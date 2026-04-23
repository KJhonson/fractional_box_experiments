function [x,sol]=findif(i,a,b,n);
%
% [x,sol]=findif(i,a,b,n)
% solves -u''=f, u(a)=exact(a), u(b)=exact(b)
% using finite differences
% n=number of subintervals
% i selects the rhs f
% needs: "f" and "exact"
%
h=(b-a)/n;
A=(diag(2*ones(1,n-1))-diag(ones(1,n-2),-1)-diag(ones(1,n-2),1))/h/h;
x=(a:h:b)';
g=f(i,x(2:n));
sol=zeros(n+1,1);
sol(1)=exact(i,x(1)); sol(n+1)=exact(i,x(n+1));
g(1)=g(1)+sol(1)/h/h; g(n-1)=g(n-1)+sol(n+1)/h/h;
sol(2:n)=A\g;
