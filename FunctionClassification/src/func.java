import java.util.*;
import java.io.*;

public class func 
{
	public static void main(String[] args) throws IOException
	{
		Scanner scan = new Scanner(System.in);
		System.out.print("Input absolute path to dataset (.csv): ");
		String path = scan.nextLine();
		
		Scanner fin = new Scanner(new File(path));
		fin.useDelimiter(",");
		ArrayList<Double> X = new ArrayList<Double>();
		ArrayList<Double> Y = new ArrayList<Double>();
		
		while(fin.hasNext())
		{
			X.add(Double.parseDouble(fin.next()));
			Y.add(Double.parseDouble(fin.next()));
		}
		System.out.println(X);
		System.out.println(Y);
		
	}
}
