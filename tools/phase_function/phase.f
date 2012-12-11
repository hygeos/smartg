c###########################################################"
c      programme pour generer une fonction de phase directement 
c      utilisable par le code de Monte CArlo a partir des 
c       sorties des SOS (fichier Trace_AEROSOLS, utilisation des alpha
c       beta, gamma, delata et zeta)
c       date: 26/10/2004
c       HYGEOS - Dominique Jolivet
c##############################################################


        character*3 ecrit
      double precision rmu, somP,somQ,somU,rtheta
        double precision rconv
        double precision RPAR1,RPAR2,RPAR3,RPAR4
      double precision P1(-1:1500,1:72001),P2(0:1500,1:72001)
      double precision alpha(0:1500),beta(0:1500),rgamma(0:1500)
        double precision delta(0:1500)

c########################################################
c       lecture des beta, etc ...
        rconv=(acos(-1.D+00))/180.D+00
        write(6,*) rconv
        write(1,*) '#'
c   On  lit ici le fichier ResultGRANU.txt
c   de sortie des SOS

      read(5,*) 
      read(5,*) 
      read(5,*) 
c   Nombre de alpha, beta ...  SOS_OS_NB dans SOS.h
      do i=0,1000
      read(5,*) alpha(i),beta(i),rgamma(i),delta(i)
c      read(5,*) ecrit,alpha(i),beta(i),rgamma(i),delta(i)
c        write(6,*) 'i= ',i
      end do

        open(1,file='pf.txt')
c       pf.txt est le fichier d'entrée pour le Monte Carlo
c       Theta, Stoke1,Stoke2, Stoke3, Stoke4
c###################################################

c####################################################
c       calcul des polynomes de Legendre d'ordre 1 et d'odre 2

c       Initialisation de P1 et P2

        do i=1,72001
        rtheta=rconv*(dble(i)-1.D0)/400.D0
        rmu=dcos(rtheta)
        P1(-1,i)=0.D0
        P1(0,i)=1.D0
        P2(0,i)=0.D0
        P2(1,i)=0.D0
        P2(2,i)=(3.D0*(1.D0-rmu**2))/(2.D0*dsqrt(6.D0))
        end do

        write(6,*) P2(2,1)

c       fin de l'initialisation

c       calcul des ordres superieur pour P1 et P2

        do i=1,72001
        rtheta=rconv*(dble(i)-1.D0)/400.D0
        rmu=dcos(rtheta)
        do k=0,999
        P1(k+1,i)=((2.D0*dble(k)+1.D0)*rmu*P1(k,i)-dble(k)
     &  *P1(k-1,i))/
     &  (dble(k)+1.D0)
        if (k.ge.2) then
        RPAR1=((2.*dble(k))+1.D0)/dsqrt((dble(k)+3.D0)*
     &  (dble(k)-1.D0))
        RPAR2=rmu*P2(k,i)
        RPAR3=(dsqrt((dble(k)+2.D0)*(dble(k)-2.D0)))/
     &  (2.D0*dble(k)+1.D0)
c        RPAR4=rmu*P2(k-1,i)
        RPAR4=P2(k-1,i)
        P2(k+1,i)=RPAR1*(RPAR2-RPAR3*RPAR4)
        endif
        end do
        end do

c       fin du calcul des polynomes de Legendre
c#################################################


c#########################################################


        do i=1,72001
        somP=0.D0
        somQ=0.D0
        somU=0.D0
        somZ=0.D0
        do k=0,999
        somP=somP+beta(k)*P1(k,i)
        somQ=somQ+rgamma(k)*P2(k,i)
        somU=somU+delta(k)*P1(k,i)
        end do
        write(6,100) (real(i)-1.)/400.,somP,somQ,somU
        write(1,100) (real(i)-1.)/400.,(somP+somQ)/2.,
     &  (somP-somQ)/2.,somU,somZ
        end do

100     format(E18.8,2x,D20.11,2x,D20.11,2x,D20.11,2x,D20.11)


        end
        






