"""
Lote 2b / B11 — login do YouTube sem recuperação nem limite de tempo.

  • creds.refresh(Request()) levantava RefreshError (invalid_grant) sem
    tratamento. O token .pickle continuava no disco, então TODO upload
    seguinte refazia o mesmo refresh e falhava — sem caminho de
    reautorização. Apps OAuth em modo "Testing" (o caso de projeto pessoal
    com o escopo youtube.upload) recebem refresh tokens de 7 dias: o YouTube
    parava de funcionar depois de uma semana.
  • Sem token, InstalledAppFlow.run_local_server(port=0) espera o navegador
    SEM timeout — no modo automático, bloqueava a thread do canal.
  • Token .pickle corrompido: toda tentativa quebrava no pickle.load.
  • O token ficava em Path.home()/AppData/Roaming/ClipBot, e não no APPDATA
    como o resto do app (o isolamento dos testes não cobria esse caminho).
  • Sem o pacote google-auth-oauthlib, a mensagem culpava outro pacote.

Contrato: falha de login vira YouTubeAuthError com instrução clara; token
revogado ou corrompido é descartado para a próxima tentativa reautorizar;
a autorização interativa tem prazo.

Isolamento: Path.home() e APPDATA apontam para tmp_path — nenhum teste toca o
token real do usuário, nem com o código antigo.
"""

import pickle
import sys
import types

import pytest
from google.auth.exceptions import RefreshError


class FakeCreds:
    """Credenciais do google-auth, no que o uploader usa."""
    def __init__(self, valid=True, expired=False, refresh_token="rt", refresh_error=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_error = refresh_error
        self.refreshed = False

    def refresh(self, request):
        if self.refresh_error:
            raise RefreshError(self.refresh_error)
        self.refreshed = True
        self.valid = True


@pytest.fixture
def yt(tmp_path, monkeypatch):
    """Uploader com home/APPDATA isolados e google-auth-oauthlib falso."""
    from pathlib import Path
    import googleapiclient.discovery
    import uploader

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    flows = []

    class FakeFlow:
        def __init__(self):
            self.kwargs = None
            self.result = FakeCreds()
            self.error = None

        @classmethod
        def from_client_secrets_file(cls, path, scopes):
            f = cls()
            flows.append(f)
            return f

        def run_local_server(self, **kw):
            self.kwargs = kw
            if self.error:
                raise self.error
            return self.result

    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.InstalledAppFlow = FakeFlow
    pkg = types.ModuleType("google_auth_oauthlib")
    pkg.flow = flow_mod
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", pkg)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_mod)

    built = []
    monkeypatch.setattr(googleapiclient.discovery, "build",
                        lambda *a, **kw: built.append(kw.get("credentials")) or "SERVICE")

    secrets = tmp_path / "client_secret.json"
    secrets.write_text("{}", encoding="utf-8")
    cfg = {"upload": {"youtube": {
        "credentials_file": str(secrets), "default_title": "t", "default_description": "d",
        "category_id": "20", "privacy": "public", "made_for_kids": False,
    }}}
    up = uploader.YouTubeUploader(cfg)
    token = tmp_path / "appdata" / "ClipBot" / "youtube_token.pickle"
    return types.SimpleNamespace(up=up, mod=uploader, flows=flows, built=built,
                                 token=token, tmp=tmp_path)


def _save_token(path, creds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(creds))


# ── Token revogado / expirado ───────────────────────────────────────────────

def test_revoked_refresh_token_asks_for_reauthorization(yt):
    _save_token(yt.token, FakeCreds(valid=False, expired=True,
                                    refresh_error="invalid_grant: Token has been expired or revoked."))

    with pytest.raises(yt.mod.YouTubeAuthError) as e:
        yt.up._get_service()

    assert "reautoriz" in str(e.value).lower()


def test_revoked_token_is_discarded_so_the_next_attempt_reauthorizes(yt):
    _save_token(yt.token, FakeCreds(valid=False, expired=True, refresh_error="invalid_grant"))

    with pytest.raises(yt.mod.YouTubeAuthError):
        yt.up._get_service()

    assert not yt.token.exists(), "token revogado ficou no disco: toda tentativa falha igual"


def test_corrupt_token_file_leads_to_reauthorization(yt):
    yt.token.parent.mkdir(parents=True, exist_ok=True)
    yt.token.write_bytes(b"isto nao e um pickle")

    assert yt.up._get_service() == "SERVICE"
    assert yt.flows, "token corrompido deveria levar à reautorização"


# ── Autorização interativa ──────────────────────────────────────────────────

def test_interactive_authorization_has_a_deadline(yt):
    yt.up._get_service()

    kw = yt.flows[0].kwargs
    assert kw.get("timeout_seconds"), "run_local_server sem timeout: espera o navegador para sempre"
    assert kw["timeout_seconds"] <= 600


def test_unfinished_authorization_becomes_an_auth_error(yt, monkeypatch):
    """O que o run_local_server faz quando o prazo acaba sem o navegador voltar."""
    import uploader
    real = sys.modules["google_auth_oauthlib.flow"].InstalledAppFlow.from_client_secrets_file

    def expiring(path, scopes):
        f = real(path, scopes)
        f.error = ValueError("authorization_response must be set (timeout)")
        return f
    monkeypatch.setattr(sys.modules["google_auth_oauthlib.flow"].InstalledAppFlow,
                        "from_client_secrets_file", staticmethod(expiring))

    with pytest.raises(uploader.YouTubeAuthError):
        yt.up._get_service()


# ── Onde o token mora ───────────────────────────────────────────────────────

def test_token_lives_in_appdata_like_the_rest_of_the_app(yt):
    yt.up._get_service()

    assert yt.token.exists(), "token não foi salvo em APPDATA/ClipBot"
    assert not (yt.tmp / "home" / "AppData" / "Roaming" / "ClipBot" / "youtube_token.pickle").exists()


def test_missing_oauth_package_is_named_correctly(yt, monkeypatch):
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", None)

    with pytest.raises(yt.mod.YouTubeAuthError) as e:
        yt.up._get_service()

    assert "google-auth-oauthlib" in str(e.value)


def test_upload_logs_the_reauthorization_instruction(yt, caplog):
    _save_token(yt.token, FakeCreds(valid=False, expired=True, refresh_error="invalid_grant"))

    with caplog.at_level("ERROR"):
        ok = yt.up.upload(yt.tmp / "clip.mp4", title="X", description="d")

    assert ok is False
    assert any("reautoriz" in r.getMessage().lower() for r in caplog.records)


# ── Comportamento preservado ────────────────────────────────────────────────

def test_valid_token_is_reused_without_new_authorization(yt):
    creds = FakeCreds(valid=True)
    _save_token(yt.token, creds)

    assert yt.up._get_service() == "SERVICE"
    assert yt.flows == []


def test_expired_token_with_working_refresh_is_renewed(yt):
    _save_token(yt.token, FakeCreds(valid=False, expired=True))

    assert yt.up._get_service() == "SERVICE"
    assert yt.flows == []
    assert pickle.loads(yt.token.read_bytes()).valid is True
