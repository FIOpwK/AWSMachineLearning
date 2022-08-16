import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.net.URL;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class UrlSigner {
    private static String keyString = "";

    private static String urlString = "";

    // This variable stores the binary key, which is computed from the string (Base64) key
    private static byte[] key;

    public static void main(String[] args) throws IOException, InvalidKeyException, NoSuchAlgorithmException, URISyntaxException {

        BufferedReader input - new BufferedReader(new InputStreamReader(System.in));
        String inputUrl, inputKey = null;

        // For testing purposes, allow user input for the URL.
        // If no input is entered, use the static URL defined above.
        System.out.println("Enter the URL (must be URL-encoded) to sign: ");
        inputUrl = input.readLine();
        if (inputUrl.equals("")) {
            inputUrl = urlString;
        }

        // Conver the string to a URL so we can parse it
        URL url = new URL(inputUrl);

        // Allow user to input private key or use static key defined above
        System.out.println("Enter the Private key to sign the URL: ");
        inputKey = input.readLine();
        if (inputKey.equals("")) {
            inputKey = keyString;
        }

        UrlSigner signer = new UrlSigner(inputKey);
        String request = signer.signRequest(url.getPath(), url.getQuery());

        System.out.println("Signed URL :" + url.getProtocol() + "://" + url.getHost() + request);

    }

    public UrlSigner(String keyString) throws IOException {
        // Convert the key from 'web safe' base 64 to binary
        keyString = keyString.replace('-', '+');
        keyString = keyString.replace('_', '/');
        System.out.println("Key: " + keyString);

        // Base64 is JDK 1.8 only - older version may need to use Apache Commons
        this.key = Base64.getDecoder().decode(keyString);
    }

    public String signRequest(String path, String query) throws NoSuchAlgorithmException, InvalidKeyException, UnsupportedEncodingException, URISyntaxException {

        // Retrieve the proper URL components to sign
        String resource = path + '?' + query;

        // Get an HMAC-SHA1 signing key from the raw key bytes
        SecretKeySpec sha1Key = new SecretKeySpec(key, "HmacSHA1");

        // Get Mac instance and initialize with HMAC-SHA1 key
        Mac mac = Mac.getInstance("HmacSHA1");
        mac.init(sha1Key);

        // compute the binary signature for the request
        byte[] signBytes = mac.doFinal(resource.getBytes());

        // base 64 encode the binary signature
        String signature = Base64.getEncoder().encodeToString(signBytes);

        // conver the signature to 'web safe' base 64
        signature = signature.replace('+', '-');
        signature = signature.replace('/','_');

        // return resource and signature
        return resource + "&signature=" + signature;
    }
}