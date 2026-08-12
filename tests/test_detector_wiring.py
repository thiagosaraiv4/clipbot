"""
Regressão P1-4 — chat IRC cross-channel.

Bug: MomentDetector.__init__ cria TwitchChatDetector baseado apenas em
     config["streams"]["twitch"]["enabled"], sem considerar QUAL plataforma
     a thread está monitorando. Threads de Kick/YouTube abrem IRC no
     primeiro canal Twitch da lista (fallback channels[0]).

Efeito: momentos de chat de um streamer da Twitch são atribuídos a um canal
        do Kick/YouTube; conexões IRC duplicadas ao mesmo canal.
Evidência em produção: duas conexões simultâneas a #baiano com 3 canais Twitch
                       e 3 Kick configurados.
"""

import copy

from conftest import detection_config


def _cfg_for_thread(platform, channel, **kw):
    """Reproduz o que _monitor_platform monta antes de instanciar o detector."""
    cfg = copy.deepcopy(detection_config(**kw))
    cfg["streams"].setdefault(platform, {"enabled": True, "channels": [channel]})
    cfg["streams"][platform]["channel"] = channel
    cfg["streams"][platform]["channel_id"] = channel
    return cfg


def test_kick_thread_has_no_twitch_chat_detector():
    """P1-4 (FALHA HOJE): canal Kick não pode abrir IRC da Twitch."""
    from moment_detector import MomentDetector

    cfg = _cfg_for_thread("kick", "jonvlogs", kick_enabled=True)
    md = MomentDetector(cfg, lambda: 0.0)

    connected_to = getattr(md.chat, "channel", None)
    assert md.chat is None, (
        f"thread do Kick criou TwitchChatDetector conectando em "
        f"'{connected_to}' — momentos do canal errado contaminam o clip"
    )


def test_youtube_thread_has_no_twitch_chat_detector():
    """P1-4 (FALHA HOJE): idem para YouTube."""
    from moment_detector import MomentDetector

    cfg = _cfg_for_thread("youtube", "https://www.youtube.com/watch?v=ABC")
    cfg["streams"]["youtube"]["enabled"] = True
    md = MomentDetector(cfg, lambda: 0.0)

    assert md.chat is None


def test_twitch_thread_uses_its_own_channel():
    """
    P1-4 (FALHA HOJE): a thread do 2º canal Twitch deve ouvir o SEU canal,
    não channels[0].
    """
    from moment_detector import MomentDetector

    cfg = _cfg_for_thread("twitch", "https://www.twitch.tv/kosky")
    md = MomentDetector(cfg, lambda: 0.0)

    assert md.chat is not None, "canal Twitch perdeu o detector de chat"
    assert md.chat.channel == "kosky", (
        f"IRC conectaria em '{md.chat.channel}' em vez de 'kosky' "
        "(fallback para channels[0])"
    )


def test_twitch_chat_can_be_disabled_by_config():
    """INVARIANTE: desligar chat_flood+keywords continua desativando o detector."""
    from moment_detector import MomentDetector

    cfg = _cfg_for_thread("twitch", "https://www.twitch.tv/baiano")
    cfg["detection"]["chat_flood"]["enabled"] = False
    cfg["detection"]["keywords"]["enabled"] = False
    md = MomentDetector(cfg, lambda: 0.0)

    assert md.chat is None
