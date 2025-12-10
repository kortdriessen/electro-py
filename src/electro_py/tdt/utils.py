from probeinterface import Probe
import numpy as np

probe_params = {}
probe_params["acr_style"] = {}
probe_params["acr_style"]["nchans"] = 16
probe_params["acr_style"]["probe_spacing"] = 50
probe_params["acr_style"]["contact_radius"] = 7.5
probe_params["acr_style"]["channel_ids"] = np.arange(1, 17)
probe_params["acr_style"]["shape"] = "circle"
probe_params["acr_style"]["polygon"] = [
    (-10, 50),
    (0, 0),
    (10, 50),
    (40, 850),
    (-40, 850),
]


def set_NNX_probe_info_on_SIrec(si_recording, probe_style: str = "acr_style"):
    probe_spacing = probe_params[probe_style]["probe_spacing"]

    CONTACT_RADIUS = probe_params[probe_style]["contact_radius"]
    POLYGON = probe_params[probe_style]["polygon"]

    # create the probe object and set assign it to the concatenated recording
    assert len(si_recording.get_channel_ids()) == probe_params[probe_style]["nchans"]
    nchans = probe_params[probe_style]["nchans"]

    # Contacts
    positions = np.zeros((nchans, 2))
    for i in range(nchans):
        x = 0
        y = (
            nchans - (i + 1)
        ) * probe_spacing + probe_spacing  # Invert here because channel 1 is most superficial and I'm getting lost with prb.set_contacts_ids() etc and don't want to invert stuff
        positions[i] = x, y

    prb = Probe(ndim=2, si_units="um")
    prb.set_contacts(
        positions=positions,
        shapes=probe_params[probe_style]["shape"],
        shape_params={"radius": CONTACT_RADIUS},
    )

    # Geometry
    prb.set_planar_contour(POLYGON)
    prb.set_device_channel_indices(np.arange(nchans))

    si_recording.set_probe(prb, in_place=True)
    return si_recording
