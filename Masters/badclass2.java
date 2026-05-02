import java.util.*;
import java.io.*;
import java.net.*;
import java.sql.*;
import java.math.*;
import javax.crypto.*;
import java.security.*;

public class ApocalypseClass extends Thread {

    private int a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z;

    // =========================
    // CHAOTIC METHODS
    // =========================
    public void m1(){ m2(); m3(); m4(); m5(); recursive(); }
    public void m2(){ m1(); m6(); m7(); m8(); }
    public void m3(){ m1(); m9(); m10(); m11(); }
    public void m4(){ m2(); m12(); m13(); m14(); }
    public void m5(){ m3(); m15(); m16(); m17(); }
    public void m6(){ m4(); m18(); m19(); }
    public void m7(){ m5(); m20(); m21(); }
    public void m8(){ m6(); m22(); m23(); }
    public void m9(){ m7(); m24(); m25(); }
    public void m10(){ m8(); m26(); m27(); }

    public void m11(){ m12(); m13(); m14(); }
    public void m12(){ m15(); m16(); m17(); }
    public void m13(){ m18(); m19(); m20(); }
    public void m14(){ m21(); m22(); m23(); }
    public void m15(){ m24(); m25(); m26(); }
    public void m16(){ m27(); m28(); m29(); }
    public void m17(){ m30(); m31(); m32(); }
    public void m18(){ m33(); m34(); m35(); }
    public void m19(){ m36(); m37(); m38(); }
    public void m20(){ m39(); m40(); m1(); }

    // =========================
    // RECURSION (🔥 BAD)
    // =========================
    public void recursive(){
        recursive(); // infinite recursion
    }

    // =========================
    // FAKE COMPLEX LOGIC
    // =========================
    public void insaneLogic(){
        for(int i=0;i<1000;i++){
            try{
                URL url = new URL("http://example.com");
                BufferedReader in = new BufferedReader(
                    new InputStreamReader(url.openStream())
                );

                String input;
                while((input = in.readLine()) != null){
                    System.out.println(input);
                    m1(); m2(); m3(); m4(); m5();
                }

                in.close();

            } catch(Exception e){
                e.printStackTrace();
                m6(); m7(); m8(); m9();
            }
        }
    }

    // =========================
    // RANDOM CROSS CALLS
    // =========================
    public void chaos(){
        for(int i=0;i<300;i++){
            m1(); m5(); m10(); m15(); m20();
            m3(); m7(); m11(); m14(); m18();
            m2(); m6(); m9(); m13(); m17();
        }
    }

    // =========================
    // THREAD OVERRIDE (UNUSED)
    // =========================
    @Override
    public void run(){
        insaneLogic();
        chaos();
        recursive();
    }
}