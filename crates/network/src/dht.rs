//! Bounded Kademlia-style routing and local DHT record storage.
//!
//! This module contains deterministic data structures only. Transport, request
//! correlation, signatures, and policy remain in `network::NetworkHandle` so
//! the DHT cannot become a second networking stack.

use intelligence_protocol::{DhtKey, DhtNamespace, DhtRecord, NodeId, SignedAnnouncement};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashSet},
    net::IpAddr,
};

pub const DEFAULT_K: usize = 20;
pub const DEFAULT_REPLACEMENT_CACHE: usize = 32;
pub const DEFAULT_MAX_RECORDS: usize = 2_048;
pub const DEFAULT_MAX_RECORDS_PER_KEY: usize = 8;
const MAX_CONTACTS_PER_SOURCE_PREFIX: usize = 4;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DhtTableStats {
    pub bucket_count: usize,
    pub populated_buckets: usize,
    pub contacts: usize,
    pub replacement_contacts: usize,
    pub protected_contacts: usize,
}

#[derive(Clone, Debug)]
pub struct DhtTable {
    local: NodeId,
    k: usize,
    replacement_limit: usize,
    buckets: Vec<Vec<SignedAnnouncement>>,
    replacements: Vec<SignedAnnouncement>,
    protected: Vec<SignedAnnouncement>,
}

impl DhtTable {
    pub fn new(local: NodeId, k: usize, replacement_limit: usize) -> Self {
        Self {
            local,
            k: k.max(1),
            replacement_limit,
            buckets: vec![Vec::new(); 256],
            replacements: Vec::new(),
            protected: Vec::new(),
        }
    }

    pub fn insert(&mut self, contact: SignedAnnouncement, now: u64) -> bool {
        let id = contact.record.node_id;
        if id == self.local || contact.record.expires_at < now {
            return false;
        }
        let index = bucket_index(self.local, id);
        let bucket = &mut self.buckets[index];
        if let Some(existing) = self
            .protected
            .iter_mut()
            .find(|item| item.record.node_id == id)
        {
            if fresher(&existing.record, &contact.record) {
                *existing = contact;
                return true;
            }
            return false;
        }
        if let Some(existing) = bucket.iter_mut().find(|item| item.record.node_id == id) {
            if fresher(&existing.record, &contact.record) {
                *existing = contact;
                return true;
            }
            return false;
        }
        if bucket.len() < self.k {
            let source = address_prefix(&contact);
            let same_source = bucket
                .iter()
                .filter(|item| address_prefix(item) == source)
                .count();
            if same_source < MAX_CONTACTS_PER_SOURCE_PREFIX {
                bucket.push(contact);
                return true;
            }
            return self.add_replacement(contact);
        }
        let source = address_prefix(&contact);
        let same_source = bucket
            .iter()
            .filter(|item| address_prefix(item) == source)
            .count();
        if same_source < self.k.div_ceil(2)
            && let Some(victim) = bucket
                .iter()
                .enumerate()
                .filter(|(_, item)| address_prefix(item) != source)
                .min_by_key(|(_, item)| item.record.announced_at)
                .map(|(index, _)| index)
        {
            bucket[victim] = contact;
            return true;
        }
        self.add_replacement(contact)
    }

    pub fn protect(&mut self, contact: SignedAnnouncement, now: u64) -> bool {
        if contact.record.node_id == self.local || contact.record.expires_at < now {
            return false;
        }
        if let Some(existing) = self
            .protected
            .iter_mut()
            .find(|item| item.record.node_id == contact.record.node_id)
        {
            if fresher(&existing.record, &contact.record) {
                *existing = contact;
                return true;
            }
            return false;
        }
        if self.protected.len() >= 16 {
            self.protected.sort_by_key(|item| item.record.expires_at);
            self.protected.remove(0);
        }
        self.protected.push(contact);
        true
    }

    pub fn expire(&mut self, now: u64) {
        for bucket in &mut self.buckets {
            bucket.retain(|contact| contact.record.expires_at >= now);
        }
        self.replacements
            .retain(|contact| contact.record.expires_at >= now);
        self.protected
            .retain(|contact| contact.record.expires_at >= now);
    }

    pub fn closest(&self, target: DhtKey, limit: usize, now: u64) -> Vec<SignedAnnouncement> {
        let mut values = self
            .buckets
            .iter()
            .flatten()
            .chain(self.replacements.iter())
            .chain(self.protected.iter())
            .filter(|contact| contact.record.expires_at >= now)
            .cloned()
            .collect::<Vec<_>>();
        values.sort_by_key(|contact| xor_distance(contact.record.node_id, NodeId(target.0)));
        values.dedup_by_key(|contact| contact.record.node_id);
        values.truncate(limit);
        values
    }

    /// Return a bounded set with source-prefix diversity before pure XOR
    /// closeness. This is a routing preference, not a trust score: a known
    /// peer cannot monopolize a bucket merely because it has good history.
    pub fn closest_diverse(
        &self,
        target: DhtKey,
        limit: usize,
        now: u64,
    ) -> Vec<SignedAnnouncement> {
        let mut values = self.closest(target, usize::MAX, now);
        let mut selected = Vec::with_capacity(limit);
        let mut sources = HashSet::new();
        for contact in values.iter() {
            if selected.len() >= limit {
                break;
            }
            if sources.insert(address_prefix(contact)) {
                selected.push(contact.clone());
            }
        }
        for contact in values.drain(..) {
            if selected.len() >= limit {
                break;
            }
            if !selected
                .iter()
                .any(|existing| existing.record.node_id == contact.record.node_id)
            {
                selected.push(contact);
            }
        }
        selected
    }

    pub fn all_contacts(&self, now: u64) -> Vec<SignedAnnouncement> {
        self.closest(DhtKey(self.local.0), usize::MAX, now)
    }

    fn add_replacement(&mut self, contact: SignedAnnouncement) -> bool {
        if let Some(existing) = self
            .replacements
            .iter_mut()
            .find(|item| item.record.node_id == contact.record.node_id)
        {
            if fresher(&existing.record, &contact.record) {
                *existing = contact;
                return true;
            }
            return false;
        }
        if self.replacement_limit == 0 {
            return false;
        }
        self.replacements.push(contact);
        if self.replacements.len() > self.replacement_limit {
            self.replacements.remove(0);
        }
        false
    }

    pub fn stats(&self) -> DhtTableStats {
        DhtTableStats {
            bucket_count: self.buckets.len(),
            populated_buckets: self
                .buckets
                .iter()
                .filter(|bucket| !bucket.is_empty())
                .count(),
            contacts: self.buckets.iter().map(Vec::len).sum(),
            replacement_contacts: self.replacements.len(),
            protected_contacts: self.protected.len(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DhtRecordStats {
    pub keys: usize,
    pub records: usize,
    pub expired_records: usize,
}

#[derive(Clone, Debug)]
pub struct DhtStore {
    records: BTreeMap<(DhtNamespace, DhtKey), Vec<DhtRecord>>,
    max_records: usize,
    max_records_per_key: usize,
}

impl DhtStore {
    pub fn new(max_records: usize, max_records_per_key: usize) -> Self {
        Self {
            records: BTreeMap::new(),
            max_records: max_records.max(1),
            max_records_per_key: max_records_per_key.max(1),
        }
    }

    pub fn put(&mut self, record: DhtRecord, now: u64) -> bool {
        if record.expires_at < now {
            return false;
        }
        let key = (record.namespace, record.key);
        let values = self.records.entry(key).or_default();
        if let Some(existing) = values.iter_mut().find(|item| item.owner == record.owner) {
            if existing.sequence >= record.sequence {
                return false;
            }
            *existing = record;
            return true;
        }
        if values.len() >= self.max_records_per_key {
            values.sort_by_key(|item| (item.sequence, item.expires_at));
            if values
                .first()
                .is_some_and(|item| item.sequence >= record.sequence)
            {
                return false;
            }
            values.remove(0);
        }
        values.push(record);
        self.enforce_total_bound(now);
        true
    }

    pub fn get(&mut self, namespace: DhtNamespace, key: DhtKey, now: u64) -> Vec<DhtRecord> {
        if let Some(values) = self.records.get_mut(&(namespace, key)) {
            values.retain(|record| record.expires_at >= now);
            let mut result = values.clone();
            result.sort_by_key(|record| std::cmp::Reverse(record.sequence));
            return result;
        }
        Vec::new()
    }

    /// Prefer one provider from each owner prefix before filling the bounded
    /// result. Provider identity is not proof of independence; this only
    /// limits a correlated flood from crowding every response slot.
    pub fn get_diverse(
        &mut self,
        namespace: DhtNamespace,
        key: DhtKey,
        now: u64,
        limit: usize,
    ) -> Vec<DhtRecord> {
        let mut values = self.get(namespace, key, now);
        let mut selected = Vec::with_capacity(limit);
        let mut prefixes = HashSet::new();
        for record in values.iter() {
            if selected.len() >= limit {
                break;
            }
            if prefixes.insert(record.owner.0[..4].to_vec()) {
                selected.push(record.clone());
            }
        }
        for record in values.drain(..) {
            if selected.len() >= limit {
                break;
            }
            if !selected.iter().any(|item| item.owner == record.owner) {
                selected.push(record);
            }
        }
        selected
    }

    pub fn expire(&mut self, now: u64) {
        self.records.retain(|_, values| {
            values.retain(|record| record.expires_at >= now);
            !values.is_empty()
        });
    }

    pub fn records(&self, now: u64) -> Vec<DhtRecord> {
        self.records
            .values()
            .flatten()
            .filter(|record| record.expires_at >= now)
            .cloned()
            .collect()
    }

    pub fn stats(&self, now: u64) -> DhtRecordStats {
        DhtRecordStats {
            keys: self.records.len(),
            records: self
                .records
                .values()
                .flatten()
                .filter(|record| record.expires_at >= now)
                .count(),
            expired_records: self
                .records
                .values()
                .flatten()
                .filter(|record| record.expires_at < now)
                .count(),
        }
    }

    fn enforce_total_bound(&mut self, now: u64) {
        self.expire(now);
        while self.records.values().map(Vec::len).sum::<usize>() > self.max_records {
            let Some((key, owner)) = self
                .records
                .iter()
                .flat_map(|(key, values)| values.iter().map(move |record| (key, record)))
                .min_by_key(|(_, record)| (record.sequence, record.expires_at))
                .map(|(key, record)| (*key, record.owner))
            else {
                break;
            };
            if let Some(values) = self.records.get_mut(&key) {
                if let Some(index) = values.iter().position(|record| record.owner == owner) {
                    values.remove(index);
                }
                if values.is_empty() {
                    self.records.remove(&key);
                }
            }
        }
    }
}

pub fn xor_distance(left: NodeId, right: NodeId) -> [u8; 32] {
    let mut result = [0u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = left.0[index] ^ right.0[index];
    }
    result
}

fn bucket_index(left: NodeId, right: NodeId) -> usize {
    let distance = xor_distance(left, right);
    for (index, byte) in distance.iter().enumerate() {
        if *byte != 0 {
            return 255usize.saturating_sub(index * 8 + byte.leading_zeros() as usize);
        }
    }
    0
}

fn address_prefix(contact: &SignedAnnouncement) -> u128 {
    let Some(address) = contact.record.addresses.first() else {
        return 0;
    };
    let Ok(address) = address.parse::<std::net::SocketAddr>() else {
        return 0;
    };
    match address.ip() {
        IpAddr::V4(value) => u32::from(value) as u128 >> 8,
        IpAddr::V6(value) => u128::from(value) >> 64,
    }
}

fn fresher(
    previous: &intelligence_protocol::PeerRecord,
    next: &intelligence_protocol::PeerRecord,
) -> bool {
    next.announced_at > previous.announced_at
        || (next.announced_at == previous.announced_at && next.expires_at >= previous.expires_at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use intelligence_protocol::{Capability, CapabilityEvidence, ResourceLimits};

    fn contact(value: u8, announced_at: u64) -> SignedAnnouncement {
        let public_key = [value; 32];
        let record = intelligence_protocol::PeerRecord {
            node_id: NodeId::from_public_key(&public_key),
            public_key,
            addresses: vec![format!("198.51.100.{}:4000", value.max(1))],
            capabilities: vec![Capability {
                name: "test".to_string(),
                version: 1,
                model: None,
                resources: ResourceLimits {
                    max_input_bytes: 1,
                    max_output_bytes: 1,
                    memory_bytes: 1,
                    cpu_millis: 1,
                },
                evidence: CapabilityEvidence::Claimed,
                expires_at: announced_at + 60,
                metadata: Vec::new(),
                compute_backends: Vec::new(),
            }],
            announced_at,
            expires_at: announced_at + 60,
            observed_latency_ms: None,
        };
        SignedAnnouncement {
            record,
            signature: vec![0; 64],
        }
    }

    #[test]
    fn routing_table_is_bounded_and_returns_closest() {
        let local = NodeId::from_bytes([0; 32]);
        let mut table = DhtTable::new(local, 2, 2);
        assert!(table.insert(contact(1, 1), 1));
        assert!(table.insert(contact(2, 1), 1));
        let closest = table.closest(DhtKey::from_bytes([1; 32]), 1, 1);
        assert_eq!(closest.len(), 1);
        assert!(table.stats().contacts <= 2 * 256);
    }

    #[test]
    fn correlated_source_contacts_are_capped_and_replaced() {
        let local = NodeId::from_bytes([0; 32]);
        let mut table = DhtTable::new(local, 8, 8);
        let first = contact(1, 1);
        let target_bucket = bucket_index(local, first.record.node_id);
        let mut same_bucket = vec![first];
        for value in 2..=u8::MAX {
            let candidate = contact(value, 1);
            if bucket_index(local, candidate.record.node_id) == target_bucket {
                same_bucket.push(candidate);
            }
            if same_bucket.len() == 5 {
                break;
            }
        }
        assert_eq!(same_bucket.len(), 5);
        for (index, candidate) in same_bucket.into_iter().enumerate() {
            assert!(table.insert(candidate, 1) || index == 4);
        }
        let stats = table.stats();
        assert!(stats.contacts <= 4);
        assert!(stats.replacement_contacts >= 1);
    }

    #[test]
    fn closest_diverse_prefers_multiple_source_prefixes() {
        let local = NodeId::from_bytes([0; 32]);
        let mut table = DhtTable::new(local, 20, 20);
        for (index, prefix) in [198_u8, 199, 200, 201].into_iter().enumerate() {
            let mut peer = contact(0x80 + index as u8, 1);
            peer.record.addresses = vec![format!("{}.51.100.10:4000", prefix)];
            assert!(table.insert(peer, 1));
        }
        let selected = table.closest_diverse(DhtKey::from_bytes([0; 32]), 4, 1);
        assert_eq!(selected.len(), 4);
        let prefixes = selected.iter().map(address_prefix).collect::<HashSet<_>>();
        assert_eq!(prefixes.len(), 4);
    }

    #[test]
    fn records_replace_by_owner_sequence_and_expire() {
        let owner = NodeId::from_bytes([1; 32]);
        let mut store = DhtStore::new(2, 2);
        let record = |sequence, expires_at| DhtRecord {
            namespace: DhtNamespace::Capability,
            key: DhtKey::from_bytes([2; 32]),
            owner,
            owner_public_key: [3; 32],
            sequence,
            expires_at,
            value: vec![sequence as u8],
            signature: vec![0; 64],
        };
        assert!(store.put(record(1, 10), 1));
        assert!(!store.put(record(1, 10), 1));
        assert!(store.put(record(2, 10), 1));
        assert_eq!(
            store.get(DhtNamespace::Capability, DhtKey::from_bytes([2; 32]), 1)[0].sequence,
            2
        );
        store.expire(11);
        assert_eq!(store.stats(11).records, 0);
    }

    #[test]
    fn provider_flood_is_bounded_and_diverse() {
        let mut store = DhtStore::new(64, 8);
        let key = DhtKey::from_bytes([2; 32]);
        for value in 1..=32_u8 {
            let accepted = store.put(
                DhtRecord {
                    namespace: DhtNamespace::Artifact,
                    key,
                    owner: NodeId::from_bytes([value; 32]),
                    owner_public_key: [value; 32],
                    sequence: 1,
                    expires_at: 100,
                    value: vec![value],
                    signature: vec![0; 64],
                },
                1,
            );
            assert_eq!(accepted, value <= 8);
        }
        assert!(store.get(DhtNamespace::Artifact, key, 1).len() <= 8);
        let diverse = store.get_diverse(DhtNamespace::Artifact, key, 1, 8);
        assert_eq!(diverse.len(), 8);
    }
}
