from hax.minitrees import TreeMaker


class LargestTriggeringSignal(TreeMaker):
    """Information on the largest trigger signal with the trigger flag set in the event

    Provides:
     - trigger_*, where * is any of the attributes of datastructure.TriggerSignal
    """
    extra_branches = ['trigger_signals*']
    __version__ = '0.0.3'

    def extract_data(self, event):
        tss = [t for t in event.trigger_signals if t.trigger]
        if not len(tss):
            # No trigger signal! This indicates an error in the trigger's signal grouping,
            # See https://github.com/XENON1T/pax/issues/344
            return dict()
        ts = tss[int(np.argmax([t.n_pulses for t in tss]))]
        return {"trigger_" + k: getattr(ts, k)
                for k in [a[0] for a in TriggerSignal.get_fields_data(TriggerSignal())]}
