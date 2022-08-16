using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Web;


namespace Name
{
    public struct GoogleSignedUrl
    {
        public static string Sign(string url, string keystring)
        {
            ASCIIEncoding encoding = new ASCIIEncoding();

            // converting key to bytes will throw an exception, need to replate '-' and '_' character first
            string usablePrivateKey = keystring.Replace("-", "+").Replace("_", "/");
            byte[] privateKeyBytes = Convert.FromBase64String(usablePrivateKey);

            Uri uri = new Uri(url);
            byte[] encodedPathAndQueryBytes = encoding.GetBytes(uri.LocalPath + uri.Query);

            // compute the hash
            HMACSHA1 algorithm = new HMACSHA1(privateKeyBytes);
            byte[] hash = algorithm.ComputeHash(encodedPathAndQueryBytes);

            // convert the bytes to string and make url-safe by replacing '+' and '/' characters
            string signature = Convert.ToBase64String(hash).Replace("+", "-").Replace("/", "_");

            // Add the signature to the existing URI
            return uri.Scheme+"://"+uri.Host+uri.LocalPath + uri.Query + "&signature=" + signature;
        }
    }

    class Program 
    {
        static void Main() 
        {
            const string keyString = "";

            const string urlString = "";

            string inputUrl = null;
            string inputKey = null;

            Console.WriteLine("Enter the URL (must be URL-encoded) to sign: ");
            inputKey = Console.ReadLine();
            if (inputKey.Length == 0) {
                inputKey = keyString;
            }

            Console.WriteLine(GoogleSignedUrl.Sign(inputUrl, inputKey));
        }
    }
    
}