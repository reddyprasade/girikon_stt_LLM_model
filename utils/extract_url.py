from urllib.parse import urlparse, parse_qs

def extract_url_text(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    query_params = parse_qs(parsed_url.query)
    call_id = query_params.get('callId', [None])[0]
    type_param = query_params.get('type', [None])[0]
    token = query_params.get('token', [None])[0]
    url_results = {'URL':url,
                    'Domain': domain,
                    'Call ID': call_id,
                    'Type': type_param,
                    'Token': token
                   }
    return url_results
