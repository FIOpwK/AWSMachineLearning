url_api = 'https://maps.googleapis.com/maps/api/staticmap?center=40.714%2c%20-73.998&zoom=12&size=400x400&key=AIzaSyChbZx3TV8ZQcUESAif4UGH2Z0Jw7cEddQ'



url_map = 'https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13&size=600x300&maptype=roadmap&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key=AIzaSyChbZx3TV8ZQcUESAif4UGH2Z0Jw7cEddQ&signature=Z6STtH8wXy_kJuYzCEyCvuQouWA='



import hashlib
import hmac
import base64
import urllib.parse as urlparse

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """
    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")
    url = urlparse.urlparse(input_url)

    # We only need to sign the path_query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # WE need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64decode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()



if __name__ == '__main__':
    input_url = input("URL to Sign: ")
    secret = input("URL signing secret: ")
    print("Signed URL: " + sign_url(input_url, secret))