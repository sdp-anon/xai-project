import java.util.*;
import java.io.*;
import java.net.*;

public class GodObject extends Thread {

    private int a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z;

    public void m1(){ m2(); m3(); m4(); m5(); m6(); }
    public void m2(){ m1(); m3(); m7(); m8(); }
    public void m3(){ m1(); m2(); m9(); m10(); }
    public void m4(){ m1(); m5(); m6(); m11(); }
    public void m5(){ m2(); m7(); m8(); m12(); }
    public void m6(){ m3(); m9(); m10(); m13(); }
    public void m7(){ m4(); m5(); m11(); }
    public void m8(){ m6(); m7(); m12(); }
    public void m9(){ m8(); m10(); m13(); }
    public void m10(){ m1(); m2(); m3(); m14(); }

    public void insaneLoop() {
        for(int i=0;i<1000;i++){
            m1(); m2(); m3(); m4(); m5();
            m6(); m7(); m8(); m9(); m10();
        }
    }

    public void networkChaos() {
        try {
            URL url = new URL("http://example.com");
            BufferedReader in = new BufferedReader(
                new InputStreamReader(url.openStream()));

            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                if(inputLine.contains("error")){
                    m1();
                } else {
                    m2();
                }
            }

            in.close();
        } catch(Exception e){
            e.printStackTrace();
        }
    }
}