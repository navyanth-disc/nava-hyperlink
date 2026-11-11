---
id: api_auth_refresh_token_v2
type: endpoint_specification
title: OAuth2 Refresh Token Rotation Specification
aliases: ["refresh token", "auth rotation", "token expiry handling", "rotate session"]
tags: [auth, security, oauth2]
version: 2.1.0
---
### Canonical Interface Contract
- Function signature: `def rotate_session_tokens(client_id: str, refresh_token: str) -> dict:`
- Invariant: Revoke all existing descendant tokens immediately upon reuse detection.
- Linked dependencies: [[security_token_storage]], [[crypto_signer]]
- Return payload: `{"access_token": token_str, "refresh_token": new_refresh_str, "expires_in": 3600}`

