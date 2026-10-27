import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc
from pathlib import Path
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(0)

def friis(pt,gt,gr,d,f):return pt+gt+gr-20*np.log10(d)-20*np.log10(f)-32.44

def gray_pam(bits):
    # Gray sequence: 00->-3,01->-1,11->+1,10->+3
    return np.where((bits[:,0]==0)&(bits[:,1]==0),-3,np.where((bits[:,0]==0)&(bits[:,1]==1),-1,np.where((bits[:,0]==1)&(bits[:,1]==1),1,3)))
def pam_decode(x):
    idx=np.digitize(x,[-2,0,2]);return np.array([[0,0],[0,1],[1,1],[1,0]],dtype=np.uint8)[idx]
def mod(bits,k):
    bits=bits.astype(np.int16)
    if k==1:return (2*bits-1).astype(complex)
    if k==2:
        b=bits.reshape(-1,2);return ((2*b[:,0]-1)+1j*(2*b[:,1]-1))/np.sqrt(2)
    if k==4:
        b=bits.reshape(-1,4);return (gray_pam(b[:,:2])+1j*gray_pam(b[:,2:]))/np.sqrt(10)
    if k==6:
        b=bits.reshape(-1,6)
        def p8(v):
            idx=v[:,0]*4+v[:,1]*2+v[:,2]
            return np.array([-7,-5,-1,-3,7,5,1,3])[idx]
        return (p8(b[:,:3])+1j*p8(b[:,3:]))/np.sqrt(42)
def demod(y,k):
    if k==1:return (y.real>=0).astype(np.uint8)
    if k==2:return np.column_stack([y.real>=0,y.imag>=0]).astype(np.uint8).ravel()
    if k==4:return np.column_stack([pam_decode(y.real*np.sqrt(10)),pam_decode(y.imag*np.sqrt(10))]).astype(np.uint8).ravel()
    if k==6:
        levels=np.array([-7,-5,-3,-1,1,3,5,7]);lut=np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,1,1],[1,0,1],[1,0,0]],dtype=np.uint8)
        a=lut[np.argmin(abs(y.real[:,None]*np.sqrt(42)-levels),axis=1)]
        b=lut[np.argmin(abs(y.imag[:,None]*np.sqrt(42)-levels),axis=1)]
        return np.column_stack([a,b]).ravel()

def awgn(symbols,k,ebn0):
    n0=1/(k*10**(ebn0/10));return symbols+np.sqrt(n0/2)*(rng.standard_normal(len(symbols))+1j*rng.standard_normal(len(symbols)))

def project21():
    d=np.linspace(.05,10,500)
    plt.figure(figsize=(8,4.5))
    for f in [900,1900,3500]:plt.plot(d,friis(43,15,0,d,f),label=f'{f} MHz')
    plt.xlabel('Distance (km)');plt.ylabel('Received power (dBm)');plt.title('Friis received power vs distance');plt.grid(alpha=.3);plt.legend();plt.tight_layout();plt.savefig(OUT/'friis.png',dpi=170);plt.close()
    print('Distance doubling loss (1->2 km):',friis(43,15,0,1,1900)-friis(43,15,0,2,1900))
    print('Frequency doubling loss (900->1800 MHz):',friis(43,15,0,1,900)-friis(43,15,0,1,1800))
    eb=np.arange(0,17,2);N=120000;results={}
    for k,name in [(1,'BPSK'),(2,'QPSK'),(4,'16-QAM'),(6,'64-QAM')]:
        bits=rng.integers(0,2,size=N,dtype=np.uint8);s=mod(bits,k);rates=[];errs=[]
        for v in eb:
            bhat=demod(awgn(s,k,v),k);err=int(np.count_nonzero(bits!=bhat));errs.append(err);rates.append(err/N)
        results[name]=(rates,errs)
    plt.figure(figsize=(8,5))
    for name in ['BPSK','QPSK','16-QAM','64-QAM']:
        r=np.array(results[name][0]);plt.semilogy(eb,np.where(r>0,r,np.nan),'o-',label=name)
    plt.semilogy(np.linspace(0,16,200),.5*erfc(np.sqrt(10**(np.linspace(0,16,200)/10))),'k--',label='BPSK theory')
    plt.ylim(1e-6,1);plt.xlabel('Eb/N0 (dB)');plt.ylabel('Bit error rate');plt.title('Monte Carlo BER, uncoded AWGN');plt.grid(True,which='both',alpha=.3);plt.legend();plt.tight_layout();plt.savefig(OUT/'ber.png',dpi=170);plt.close()
    with open(OUT/'ber_results.csv','w') as f:
        f.write('EbN0_dB,BPSK,QPSK,16QAM,64QAM\n')
        for i,v in enumerate(eb):f.write(f"{v},"+','.join(str(results[n][0][i]) for n in ['BPSK','QPSK','16-QAM','64-QAM'])+'\n')
    print('BER results',results)
    d2=np.linspace(1,10,500);plt.figure(figsize=(8,4.5))
    for n in [2,3,4]:
        pl0=20*np.log10(.1)+20*np.log10(1900)+32.44
        plt.plot(d2,pl0+10*n*np.log10(d2/.1),label=f'n={n}')
    plt.xlabel('Distance (km)');plt.ylabel('Path loss (dB)');plt.title('Log-distance model, reference 0.1 km');plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(OUT/'logdistance.png',dpi=170);plt.close()
    return eb,results

def project22():
    N=64;K=4;nsym=500;delay=8;h=np.zeros(delay+1,dtype=complex);h[0]=1;h[3]=.55*np.exp(.6j);h[8]=.4*np.exp(-.9j)
    bits=rng.integers(0,2,size=(nsym,N*K),dtype=np.uint8)
    symbols=mod(bits.ravel(),4).reshape(nsym,N)
    x=np.fft.ifft(symbols,axis=1)
    assert np.allclose(np.fft.fft(x,axis=1),symbols)
    H=np.fft.fft(h,N)
    stats=[]; const={}
    for cp in [0,2,4,6,8,10,12,16]:
        frame=np.concatenate([x[:,-cp:],x],axis=1) if cp else x
        tx=frame.ravel();rx=np.convolve(tx,h)[:len(tx)]
        window=rx.reshape(nsym,N+cp)[:,cp:]
        equalized=np.fft.fft(window,axis=1)/H[None,:]
        decoded=demod(equalized.ravel(),4).reshape(nsym,N*K)
        ber=np.count_nonzero(decoded[1:]!=bits[1:])/bits[1:].size
        mse=np.mean(abs(equalized[1:]-symbols[1:])**2)
        stats.append((cp,ber,mse))
        if cp in [0,8]:const[cp]=equalized[1:].ravel()
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,cp in zip(axs,[0,8]):
        vals=const[cp][:2500];ax.scatter(vals.real,vals.imag,s=3,alpha=.3);ax.set_title(f'CP={cp} samples');ax.set_xlabel('In-phase');ax.set_ylabel('Quadrature');ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.grid(alpha=.2);ax.set_aspect('equal')
    fig.suptitle('Equalized 16-QAM constellations: same multipath channel');fig.tight_layout();fig.savefig(OUT/'ofdm_constellations.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot([r[0] for r in stats],[r[1] for r in stats],'o-',label='BER');ax.axvline(delay,ls='--',color='gray',label='Max delay = 8 samples');ax.set_xlabel('CP length (samples)');ax.set_ylabel('Bit error rate');ax.set_title('OFDM BER vs cyclic prefix length (no noise)');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUT/'ofdm_cp_sweep.png',dpi=170);plt.close(fig)
    with open(OUT/'ofdm_results.csv','w') as f:
        f.write('cp_samples,ber,mse\n');f.writelines(f'{a},{b},{c}\n' for a,b,c in stats)
    print('OFDM CP results:',stats)
    return stats

def project23():
    # Illustrative assumptions; throughput figures are scenario inputs, NOT published standards maxima.
    data=[('Bluetooth BLE',10,1,'Low',1),('Wi-Fi',75,100,'Medium',2),('WiMAX',5000,20,'High',3)]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,r,t,p,sz in data:
        ax.scatter(r,t,s=sz*180,alpha=.65);ax.annotate(name,(r,t),xytext=(5,5),textcoords='offset points')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Assumed typical coverage radius (m)');ax.set_ylabel('Illustrative application throughput (Mbps)');ax.set_title('Range / throughput / power-class trade-off');ax.grid(True,which='both',alpha=.3);fig.tight_layout();fig.savefig(OUT/'wireless_tradeoff.png',dpi=170);plt.close(fig)
    with open(OUT/'wireless_dataset.csv','w') as f:
        f.write('technology,range_m,illustrative_throughput_mbps,power_class\n');f.writelines(f'{n},{r},{t},{p}\n' for n,r,t,p,_ in data)
    return data

if __name__=='__main__':
    project21();project22();print('Dataset:',project23())





