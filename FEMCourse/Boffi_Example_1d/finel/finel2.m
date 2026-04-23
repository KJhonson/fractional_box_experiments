function [x,sol]=finel2(i,a,b,n);
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
% 8 nodes Gauss formula
%
xx=[-.9602898565 -.7966664774 -.5255324099 -.1834346425...
     .1834346425  .5255324099  .7966664774  .9602898565];
bb1=xx/2+1/2;
bb2=-xx/2+1/2;
pp=[.1012285363 .2223810345 .3137066459 .3626837834...
    .3626837834 .3137066459 .2223810345 .1012285363];
pp=pp/sum(pp);
g=zeros(n-1,1);
for ii=1:n-1
  if ii==n/2 & i==3
    g(ii)=h^(1/4)*2;
  elseif (ii==(n-1)/2 | ii==(n+1)/2) & i==3
    g(ii)=(5/4+3^(5/4)/2-7/4)*(h/2)^(1/4);
  else
    xx1=bb1*h+x(ii);
    xx2=xx1+h;
    g(ii)=h*(sum(pp.*f(i,xx1).*bb1)+sum(pp.*f(i,xx2).*bb2));
  end
end
%
sol=zeros(n+1,1);
sol(1)=exact(i,x(1)); sol(n+1)=exact(i,x(n+1));
g(1)=g(1)+sol(1)/h; g(n-1)=g(n-1)+sol(n+1)/h;
sol(2:n)=A\g;
